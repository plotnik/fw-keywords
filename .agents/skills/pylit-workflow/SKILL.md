---
name: pylit-workflow
description: Preserve literate programming when reading, editing, or reviewing PyLit Python projects with paired .py and .py.rst files and Sphinx documentation. Leave RST synchronization and documentation builds to the user unless explicitly requested.
---

# PyLit literate programming

Use this workflow in PyLit projects. It is portable across repositories and has no project-specific paths.

## Purpose

Literate programming presents code as an article or book: explanations and executable code form a coherent account of what the program does and why. Preserve that readability and transparency. Follow the project's narrative organization, naming, and comment conventions. Explain non-obvious intent where it helps the reader, without narrating obvious operations or reorganizing unrelated code.

## Workflow

PyLit maintains programs in paired Python (`*.py`) and reStructuredText (`*.py.rst`) files. Sphinx processes the RST documents into readable project documentation.

- Read Python code (`*.py`) when useful for understanding the implementation and rationale. Its paired RST narrative (`*.py.rst`) narrative may not yet reflect recent Python changes and should be ignored. 
- Make program changes in `*.py` files, preserving the existing PyLit-compatible comment and code layout.
- Leave `*.py.rst` files unchanged during ordinary code work. Do not create, edit, or regenerate counterparts automatically.
- The user updates the corresponding RST files after Python changes and rebuilds Sphinx documentation manually. Do not run PyLit synchronization or Sphinx builds unless explicitly requested.
- Check project task and build commands before running them if they might also synchronize PyLit files or build documentation. Use code validation commands that avoid those steps.
- Temporary differences between changed Python files and their RST counterparts are expected. Do not treat them alone as a defect or unfinished work.
- Identify changed Python files in the change summary so the user can update their paired documents.

Explicit user requests to edit RST, synchronize PyLit files, or build Sphinx documentation override the manual workflow for that task.

## Reference

[PyLit documentation](http://slott56.github.io/PyLit-3/)

Consult this reference when conversion behavior or PyLit syntax matters. Routine Python edits do not require reading the manual.
