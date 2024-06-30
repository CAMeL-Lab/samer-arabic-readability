#!/bin/bash

CAMELTOOLS_DATA="$HOME/cameltools_data"

if [ ! -d "$CAMELTOOLS_DATA" ]; then
  mkdir -p "$CAMELTOOLS_DATA"
fi

export CAMELTOOLS_DATA

camel_data -i all
