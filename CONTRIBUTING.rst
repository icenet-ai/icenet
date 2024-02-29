.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/icenet-ai/icenet/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

IceNet could always use more documentation, whether as part of the
official IceNet docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/icenet-ai/icenet/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `icenet` for local development.

1. Fork the `icenet` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/icenet.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv icenet
    $ cd icenet/
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. Install development packages::

    $ pip install -r requirements_dev.txt

6. Set up pre-commit hooks to run automatically. This will run through linting checks, formatting, and pytest. It will format new code using yapf and prevent code committing that does not pass linting or testing checks until fixed::

    $ pre-commit install

7. When you're done making changes, check that your changes pass flake8 and the tests::

    $ make lint
    $ pytest

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

   Note: When committing, if pre-commit is installed, the commit might be prevented if there are problems with formatting. In this case, deal with the file(s) and commit again.

9.  Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.

Tips
----

TODO


Deploying
---------

A reminder for the maintainers on how to deploy::

$ make clean
$ make lint # Ignore black moaning at present
$ make docs
$ make install
$ make release

If anything looks really wrong, abandon and fix!
