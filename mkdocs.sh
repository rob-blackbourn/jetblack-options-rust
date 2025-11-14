#!/bin/bash

RUSTDOCFLAGS="--html-in-header katex-dollar.html" cargo doc --no-deps
