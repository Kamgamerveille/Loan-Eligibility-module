import importlib


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "sklearn",
    "streamlit",
]


def test_required_packages():
    """
    Confirm that all required packages can be imported.
    """

    for package in REQUIRED_PACKAGES:
        importlib.import_module(package)


if __name__ == "__main__":
    test_required_packages()

    print(
        "All required Real Estate project "
        "packages are installed."
    )