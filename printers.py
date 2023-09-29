# -*- python-indent-offset: 2 -*-
#!/usr/bin/env python3
from math import ceil, floor

def headerprint(string, mychar='='):
    """ Prints a centered string to divide output sections. """
    mywidth = 78
    numspaces = mywidth - len(string)
    before = int(ceil(float(mywidth-len(string))/2))
    after  = int(floor(float(mywidth-len(string))/2))
    print("\n"+before*mychar+string+after*mychar+"\n")

def valprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4f} {2}".format(string, value, unit))

def valeprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4e} {2}".format(string, value, unit))

def strprint(string, value):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}:  {1}".format(string, value))

def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)
