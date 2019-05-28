from setuptools import setup
setup(
    name="libtalley",
    version="0.6.4",
    packages=[
        'libtalley',
        'libtalley.asce7_16',
    ],
    package_data={
        'libtalley': [
            'aisc-shapes-database-v15-0-SI.csv.bz2',
            'aisc-shapes-database-v15-0-US.csv.bz2',
            'steel-materials.csv',
        ]
    },
    install_requires=[
        'numpy >= 1.15.0',
        'pandas',
        'scipy >= 1.0.0',
        'tabulate',
        'xlrd',
    ],
    python_requires=">=3.7.0",
    author="Peter Talley",
    author_email="peterctalley@gmail.com",
    description="A collection of helpful functions and doodads.",
    license="MIT",
)
