# cl-fann
Written by Mason Smith [masonium@gmail.com](mailto:masonium@gmail.com)

1. [What is cl-fann?](#whatis)
2. [Installation](#install)
3. [Usage](#usage)
4. [Limitations](#limit)
5. [TODO](#todo)

<h2 id="whatis">1. What is cl-fann?</h2>
cl-fann is a CFFI-wrapper for [libfann](http://leenissen.dk/fann/), a fast
artificial neural network. In addition, cl-fann provides a lispy interface to
libfann, providing idiomatic macros and functions for easy integration into
existing programs.  

<h2 id="install">2. Installation</h2>
cl-fann is installable via ASDF. Simply make a symbolic link to
fann.asd in your ASDF directory, and use `(asdf:oos 'asdf:load-op
:fann)` to compile and load the program. Alternatively, if you use
clbuild, you can add cl-fann to your wnpp-projects file. The repo is located
at: 

http://github.com/masonium/cl-fann.git

In addition to libfann 2.*, cl-fann requires the cffi, trivial-garbage, and alexandria libraries, all installable
via ASDF-install or clbuild. With all of the dependencies, a clbuild line would look like:

cl-fann get_git http://github.com/masonium/cl-fann.git get_trivial-garbage get_cffi get_alexandria

<h2 id="usage">3. Usage</h2>

<h2 id="limit">4. Limitations</h2>
The current working version of cl-fann is designed around version 2.0.0 of libfann. This is somewhat behind the current release version of libfann, which is 2.1.0b. I'm currently targeting 2.0.0 because this version has a binary
release on all major platforms. 

<h2 id="todo">5. TODO</h2>
* Complete function coverage (mostly complete)

* (Thorough) testing on various platforms and CL implementations

* Versioning / tagging releases in the repository

* Updating to 2.1.0b or 2.2.0, once the latter is released.
