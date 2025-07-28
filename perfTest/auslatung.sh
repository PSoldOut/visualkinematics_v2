#!/bin/bash

INTERVAL=5  # Zeit in Sekunden

while true; do
    {
        top -b -n 1 >> ~/top_out4clients_ohne_restsessions.txt
        echo
        sleep $INTERVAL
        
    }

    
done
