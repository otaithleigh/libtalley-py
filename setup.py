from setuptools import setup
setup(
    name="libtalley",
    version="0.1",
    py_modules=[
        'libtalley',
        'steeldesign',
        'fema_p695'
    ],
    packages=[
        'asce7_16',
    ],
    data_files=[
        'aisc-shapes-database-v15.0.xlsx',
        'aisc-shapes-database-v15.0.p',
    ],
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
