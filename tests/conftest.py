import pytest

from pac import ontology as ontology_module
from pac.records import Product


@pytest.fixture(scope="session")
def ontology():
    return ontology_module.load()


@pytest.fixture(scope="session")
def neckline(ontology):
    return ontology.attributes["neckline"]


@pytest.fixture(scope="session")
def sleeve(ontology):
    return ontology.attributes["sleeve_length"]


def make_product(pid, category="Tshirts", neck="Round Neck", sleeve="Short Sleeves", siblings=(),
                 description="", brand="Acme", title=None):
    return Product(
        id=pid,
        category=category,
        brand=brand,
        title=title if title is not None else f"{brand} Men Navy {category}",
        variant="",
        description=description,
        raw_labels={"Neck": neck, "Sleeve Length": sleeve},
        siblings=frozenset(siblings),
    )
