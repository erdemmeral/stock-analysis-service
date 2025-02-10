#!/bin/bash

# Start the TCP health check server in the background
python -c "
import socket
s = socket.socket()
s.bind(('', 8080))
s.listen(1)
while True:
    c, addr = s.accept()
    c.close()
" &

# Start the main application
exec python -m src.main 