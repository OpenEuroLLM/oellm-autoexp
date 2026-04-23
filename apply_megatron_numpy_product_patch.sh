#!/bin/bash

for i in $(find ./submodules/); do
    if [[ -f $i ]]; then
        sed -i 's/np.product/np.prod/' $i ;
    fi ;
done
