# -*- coding: utf-8 -*-
__author__ = 'hypo'

import numpy as np
from random import *
from collections import *

targetSeq = []

def ran01():
    r = random()
    r *= 10
    if r >= 5:
        return 1
    else:
        return 0

class coeff:
    def __init__(self, inbits):

        if inbits is None:
            self.sign = ran01()
            self.bits = []
            for i in range(9):
                self.bits.append(ran01())
        else:
            self.sign = inbits[0]
            self.bits = []
            for i in range(len(inbits)):
                if i == 0:
                    continue
                self.bits.append(inbits[i])


    def __repr__(self):
        if self.sign == 0:
            out = "0"
        else:
            out = "1"
        for bit in self.bits:
            if bit == 0:
                out = out +"0"
            else:
                out = out +"1"
        return out

    def eval(self):
        value = 0
        counter = 0
        for i in [6, 5, 4, 3, 2, 1, 0, -1, -2]:
            value = value + self.bits[counter] * pow(2,i)
            counter += 1
        if self.sign == 1:
            value *= -1
        return value


class polynomial6:

    def __init__(self, wsp):
        global targetSeq
        if wsp is None:
            self.coeffs = []
            for i in range(7):
                self.coeffs.append(coeff(None))
        else:
            self.coeffs = []
            for coef in wsp:
                self.coeffs.append(coef)
        self.fit = fitness(self, targetSeq)

    def __repr__(self):
        out = ""
        i = 0
        for wsp in self.coeffs:
            out = out + str(wsp.eval()) + "x" + str((6-i))+" "
            i=i+1
        return out

    def __cmp__(self, other):
        out = 0
        if self.fit < other.fit:
            out = 1
        else:
            if self.fit > other.fit:
                out = -1
            else:
                return 0
        return out

    def __lt__(self, other):
        if self.fit > other.fit:
            return False
        else:
            return True



    def eval(self, x):
        value = 0
        power = 1
        for i in range(7):
            value = value + power * self.coeffs[6-i].eval()
            power = power * x
        return value

def crosscoeff(c1, c2):
    outlist = []
    outlist.append(c1.sign)
    for i in range(9):
        if i+1%2 == 0:
            outlist.append(c1.bits[i])
        else:
            outlist.append(c2.bits[i])
    outc = coeff(outlist)
    return outc

def crossold(poly1, poly2):
    c1 = poly1.coeffs[0]
    c2list = []
    #print(poly1.coeffs[1].bits[0:5])
    c2list = c2list +[poly1.coeffs[1].sign]
    c2list = c2list + poly1.coeffs[1].bits[0:4]
    c2list = c2list + poly2.coeffs[1].bits[4:9]
    #print (c2list)
    c2 = coeff(c2list)
    c3 = crosscoeff(poly1.coeffs[2], poly2.coeffs[2])
    c4 = crosscoeff(poly1.coeffs[3], poly2.coeffs[3])
    c5 = coeff(None)
    c6 = coeff(None)
    c7 = coeff(None)
    outpoly = polynomial6([c1,c2,c3,c4,c5,c6,c7])
    return outpoly

def cross(poly1, poly2):
    #suppose poly1 is better
    c1= crosscoeff(poly1.coeffs[0], poly2.coeffs[0])
    c2= crosscoeff(poly1.coeffs[1], poly2.coeffs[1])
    c3= crosscoeff(poly1.coeffs[2], poly2.coeffs[2])
    c4= crosscoeff(poly1.coeffs[3], poly2.coeffs[3])
    c5= crosscoeff(poly1.coeffs[4], poly2.coeffs[4])
    c6= crosscoeff(poly1.coeffs[5], poly2.coeffs[5])
    c7= crosscoeff(poly1.coeffs[6], poly2.coeffs[6])

    outpoly = polynomial6([c1,c2,c3,c4,c5,c6,c7])
    return outpoly

def fitness(polynomial, targetSequence):
    error = 0
    for pair in targetSequence:
        error += abs((polynomial.eval(pair[0]) - pair[1]))
    return error/100
    
def mutate(poly):
    newlist = []
    for i in range(7):
        newlist.append(poly.coeffs[i])
    mutateAt = int(random()*7)
    newlist[mutateAt] = coeff(None)
    outpoly = polynomial6(newlist)
    return outpoly

previousFit100 = None
previousFit = None
withoutProgress = 0
def iteration(num):
    global population, previousFit100
    for i in range(int(pop_size/2)):
        i1 = int(random()*(pop_size/2))
        i2 = int(random()*(pop_size/2))
        if(random() < 0.5):
            nowy = crossold(population[i1], population[i2])
        else:
            nowy = cross(population[i1], population[i2])

        population[i+int(pop_size/2)] = nowy
    population.sort()
    for i in range(pop_size-5):
        if random() < 0.03:
            population[i+4] = mutate(population[i+4])
    population.sort()
    if num %100 == 0:
        print ("BEST " + str(num))
        print(population[0])
        print(population[0].fit)
        if previousFit100 is not None:
            print("IMPROVEMENT " + str(previousFit100-population[0].fit))
        previousFit100 = population[0].fit



iterations = 100001
def run():
    population.sort()
    for iterationCount in range(iterations):
        iteration(iterationCount+1)

a1 = coeff([0,0,0,0,0,0,0,1,0,0])
a2 = coeff([0,0,0,0,0,0,1,0,0,0])
a3 = coeff([0,0,0,0,0,0,1,1,0,0])
a4 = coeff([0,0,0,0,0,1,0,0,0,0])
a5 = coeff([0,0,0,0,0,1,0,1,0,0])
a6 = coeff([0,0,0,0,0,1,1,0,0,0])
a7 = coeff([0,0,0,0,0,1,1,1,0,0])
SuperPolynomial = polynomial6([a1,a2,a3,a4,a5,a6,a7])


pop_size = 1000 #


x = np.linspace(-100, 100, 100)
for i in range(100):
    y = SuperPolynomial.eval(x[i])
    pair = (x[i], y)
    targetSeq.append(pair)

population = []
for i in range(pop_size):
    population.append(polynomial6(None))

print ("TARGET POLYNOMIAL ")
print (SuperPolynomial)
print ("TARGET SEQUENCE")
print(targetSeq)
print("FITNESS OF THE TARGET POLYNOMIAL")
print(fitness(SuperPolynomial, targetSeq))

run()
x = np.linspace(-100, 100, 200)
y = population[0].eval(x)

