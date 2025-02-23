#!/bin/bash
curl -sL https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xvz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
