from setuptools import setup 

setup(name="ROP tool",
      version="0.0.1",
      description="A ROP tool",
      author="PFmai",
      author_email="pfmailard@netc.Fr",
      install_requires=["capstone", "pickle-mixin", "sklearn", "angrop", "multiprocess", "plot"]
      )