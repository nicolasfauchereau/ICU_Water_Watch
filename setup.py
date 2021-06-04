from distutils.core import setup

setup(name='ICU_Water_Watch',
    version='0.1',
    author = "Nicolas Fauchereau",
    author_email = "Nicolas.Fauchereau@niwa.co.nz",
    description = ("A set of functions for the processing of GPM-IMERG satellite rainfall and C3S MME forecasts"),
    url = "https://github.com/nicolasfauchereau/ICU_Water_Watch",
    license = "LICENSE.txt",
    long_description = open('README.md').read(),
    packages=['ICU_Water_Watch'],
)