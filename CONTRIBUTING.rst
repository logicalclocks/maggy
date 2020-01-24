How to contribute
=================

Contributions are welcome! Not familiar with the codebase yet? No problem!
There are many ways to contribute to open source projects: reporting bugs,
helping with the documentation, spreading the word and of course, adding
new features and patches.

Reporting issues
----------------

- Describe what you expected to happen.
- If possible, include a `minimal, complete, and verifiable example`_ to help
  us identify the issue. This also helps to check that the issue is not with
  your own code.
- Describe what actually happened. Include the full traceback if there was an
  exception.
- List your Python, Hopsworks and Maggy versions. If possible, check if this
  issue is already fixed in the repository.

.. _minimal, complete, and verifiable example: https://stackoverflow.com/help/mcve

Contributing Code
-----------------

Code contributions, in the form of patches or features are welcome. In order to
start developing, please follow the instructions below, to enable pre-commit_ and
ensure style and codechecks.

.. _pre-commit: https://pre-commit.com/

Python Setup
~~~~~~~~~~~~

- Fork Maggy to your GitHub account by clicking the `Fork` button.

- Clone your fork locally::

        git clone https://github.com/[username]/maggy.git
        cd maggy

- Add the upstream repository as a remote to update later::

        git remote add upstream https://github.com/logicalclocks/maggy.git
	git fetch upstream

- Create a new Python environment with your favourite environment manager, e.g. virtualenv or conda::

        python3 -m venv env
        . env/bin/activate
        # or "env\Scripts\activate" on Windows

  or with conda::

        conda create --name maggy python=3.6.8

  verify your python version - we are using Python 3.6::

        python --version

- Install Maggy in editable mode with development dependencies::

        pip install -e ".[dev]"

- Install pre-commit_ and then activate its hooks. pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. Maggy uses pre-commit to ensure code-style and code formatting through black_ and flake8_::

        pip install --user pre-commit
        pre-commit install

  Afterwards, pre-commit will run whenever you commit.

.. _pre-commit: https://pre-commit.com/
.. _flake8: https://gitlab.com/pycqa/flake8
.. _black: https://github.com/psf/black

- To run formatting and code-style separately, you can configure your IDE, such as VSCode, to use black_ and flake8_, or run them via the command line::

        flake8 maggy
        black maggy

Start coding
~~~~~~~~~~~~

- Create a branch to identify the issue or feature you would like to work on.
- Using your favorite editor, make your changes, `committing as you go`_.
- Follow `PEP8`_.
- Push your commits to GitHub and `create a pull request`_.
- Celebrate 🎉

.. _committing as you go: http://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html#commit-your-changes
.. _PEP8: https://pep8.org/
.. _create a pull request: https://help.github.com/articles/creating-a-pull-request/
