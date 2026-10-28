from collections import Counter, defaultdict

from conftest import make_product

from pac.splits import (CROSSING_SPLIT, DROPPED_LABEL, FAMILY_DISAGREEMENT, HELD_OUT_SPLIT, INSPECTED,
                        NO_USABLE_PHOTO, TEST_SPLITS, TEXT_CONFLICT, UNSUPPORTED_LABEL, assign_splits, bucket,
                        families, label_rows, usable_photos)


def test_families_join_colour_variants_transitively():
    products = [make_product(5, siblings=[7]), make_product(7, siblings=[9]), make_product(9), make_product(4)]
    family = families(products)
    assert family[5] == family[7] == family[9] == 5
    assert family[4] == 4


def test_byte_identical_photos_join_one_family_and_one_split():
    # The catalog lists some garments twice under different IDs, with no colour link (data review).
    products = [make_product(i) for i in range(1, 400)]
    hashes = {i: f"photo-{i // 2}" for i in range(1, 400)}  # 2 and 3 share a photo, 4 and 5, ...
    family = families(products, hashes)
    splits = assign_splits(products, family)
    for i in range(2, 398, 2):
        assert family[i] == family[i + 1] == i
        assert splits[i].split == splits[i + 1].split
    assert family[1] == 1


def test_buckets_are_deterministic_and_proportional():
    names = [bucket(i) for i in range(20_000)]
    assert names == [bucket(i) for i in range(20_000)]
    share = {k: v / len(names) for k, v in Counter(names).items()}
    for split, expected in {"train": 0.70, "dev": 0.10, "cal_fit": 0.05, "cal_threshold": 0.05, "test": 0.10}.items():
        assert abs(share[split] - expected) < 0.015, split


def test_a_family_never_spans_two_splits():
    products = [make_product(i, siblings=[i + 1] if i % 3 else []) for i in range(1, 600)]
    splits = assign_splits(products, families(products))
    seen = defaultdict(set)
    for row in splits.values():
        seen[row.family_id].add(row.split)
    assert all(len(s) == 1 for s in seen.values())


def test_held_out_categories_only_reach_the_held_out_split():
    products = [make_product(1, "Kurtas"), make_product(2, "Dresses"), make_product(3, "Tshirts")]
    splits = assign_splits(products, families(products))
    assert splits[1].split == splits[2].split == HELD_OUT_SPLIT
    assert splits[3].split != HELD_OUT_SPLIT


def test_families_crossing_the_split_boundary_are_excluded():
    products = [make_product(1, "Tshirts", siblings=[2]), make_product(2, "Kurtas")]
    splits = assign_splits(products, families(products))
    assert splits[1].split == splits[2].split == CROSSING_SPLIT


def test_out_of_scope_categories_are_ignored():
    products = [make_product(1, "Shorts"), make_product(2, "Tshirts")]
    assert set(assign_splits(products, families(products))) == {2}


def _rows(products, ontology):
    splits = assign_splits(products, families(products))
    return {(r.product_id, r.attribute): r for r in label_rows(products, splits, ontology)}, splits


def test_label_exclusion_reasons(ontology):
    train_like = [make_product(i, neck="Round Neck") for i in range(1000, 1100)]
    products = train_like + [
        make_product(1, neck="Round Neck", description="Navy T-shirt, has a V-neck"),
        make_product(2, neck="Halter Neck"),
        make_product(3, neck="Round Neck", siblings=[4]),
        make_product(4, neck="V-Neck"),
        make_product(5, "Kurtas", neck="Cowl Neck"),
        make_product(6, neck="Round Neck", sleeve="NA"),
    ]
    rows, _ = _rows(products, ontology)
    assert rows[(1, "neckline")].reason == TEXT_CONFLICT
    assert rows[(2, "neckline")].reason == DROPPED_LABEL
    assert rows[(3, "neckline")].reason == rows[(4, "neckline")].reason == FAMILY_DISAGREEMENT
    assert rows[(5, "neckline")].reason == UNSUPPORTED_LABEL
    assert (6, "sleeve_length") not in rows  # unlabeled, not excluded
    assert rows[(6, "neckline")].included


def test_a_dropped_value_makes_the_family_disagree(ontology):
    train_like = [make_product(i, neck="Round Neck") for i in range(1000, 1100)]
    products = train_like + [make_product(1, neck="Round Neck", siblings=[2]), make_product(2, neck="Halter Neck")]
    rows, _ = _rows(products, ontology)
    assert rows[(1, "neckline")].reason == FAMILY_DISAGREEMENT
    assert rows[(2, "neckline")].reason == DROPPED_LABEL


def test_the_variant_name_is_not_evidence(ontology):
    # The model never sees the variant name, so it cannot make the text agree or conflict.
    rows, _ = _rows([make_product(1, neck="Round Neck", variant="V-Neck Tee")], ontology)
    assert rows[(1, "neckline")].text_status == "silent"


def test_inspected_items_never_reach_a_test_set(ontology):
    products = [make_product(i) for i in range(1, 300)] + [make_product(i, "Kurtas") for i in range(300, 340)]
    splits = assign_splits(products, families(products))
    inspected = {(pid, "neckline") for pid in splits}
    rows = {r.product_id: r for r in label_rows(products, splits, ontology, None, inspected)
            if r.attribute == "neckline"}
    assert {s.split for s in splits.values()} >= {"train", "test", HELD_OUT_SPLIT}
    for pid, s in splits.items():
        assert (rows[pid].reason == INSPECTED) == (s.split in TEST_SPLITS)


def test_products_without_a_usable_photo_are_excluded(ontology):
    products = [make_product(1), make_product(2), make_product(3)]
    photos = [{"product_id": "1", "status": "ok", "width": "1080", "height": "1440"},
              {"product_id": "2", "status": "ok", "width": "150", "height": "200"},
              {"product_id": "3", "status": "missing", "width": "", "height": ""}]
    splits = assign_splits(products, families(products))
    rows = {r.product_id: r for r in label_rows(products, splits, ontology, usable_photos(photos))
            if r.attribute == "neckline"}
    assert rows[1].included
    assert rows[2].reason == rows[3].reason == NO_USABLE_PHOTO


def test_roll_up_sleeves_only_leave_the_sleeve_task(ontology):
    products = [make_product(1, "Shirts", neck="NA", sleeve="Short Sleeves",
                             description="Check shirt, has long sleeves with a mid-arm buttoned tab"),
                make_product(2, neck="Round Neck", sleeve="Short Sleeves",
                             description="Navy T-shirt with roll-up sleeves")]
    rows, _ = _rows(products, ontology)
    assert rows[(1, "sleeve_length")].reason == rows[(2, "sleeve_length")].reason == "roll_up_sleeves"
    assert rows[(2, "neckline")].included


def test_catalog_roll_up_styling_leaves_the_sleeve_task(ontology):
    rows, _ = _rows([make_product(1, sleeve="Long Sleeves", styling="Roll-Up Sleeves")], ontology)
    assert rows[(1, "sleeve_length")].reason == "roll_up_sleeves"
    assert rows[(1, "neckline")].included


def test_labels_named_only_through_the_brand_do_not_conflict(ontology):
    products = [make_product(1, neck="Round Neck", brand="U.S. Polo Assn.", title="U.S. Polo Assn. Men Navy T-shirt")]
    rows, _ = _rows(products, ontology)
    assert rows[(1, "neckline")].text_status == "silent"
