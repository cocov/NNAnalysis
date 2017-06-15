# copyright 2010 Eric Gradman
# free to use for any purpose, with or without attribution
# from an algorithm by James McNeill at
# http://playtechs.blogspot.com/2007/04/hex-grids.html

# the center of hex (0,0) is located at cartesian coordinates (0,0)

import numpy as np

# R ~ center of hex to edge
# S ~ edge length, also center to vertex
# T ~ "height of triangle"

real_R = 75. # in my application, a hex is 2*75 pixels wide
R = 2.
S = 2.*R/np.sqrt(3.)
T = S/2.
SCALE = real_R/R

# XM*X = I
# XM = Xinv
X = np.array([
  [ 0, R],
  [-S, S/2.]
])
XM = np.array([
  [1./(2.*R),  -1./S],
  [1./R,        0.  ]
])
# YM*Y = I
# YM = Yinv
Y = np.array([
  [R,    -R],
  [S/2.,  S/2.]
])
YM = np.array([
  [ 1./(2.*R), 1./S],
  [-1./(2.*R), 1./S],
])

def cartesian2hex(cp):
  """convert cartesian point cp to hex coord hp"""
  cp = np.multiply(cp, 1./SCALE)
  Mi = np.floor(np.dot(XM, cp))
  xi, yi = Mi
  i = np.floor((xi+yi+2.)/3.)

  Mj = np.floor(np.dot(YM, cp))
  xj, yj = Mj
  j = np.floor((xj+yj+2.)/3.)

  hp = i,j
  return hp

def hex2cartesian(hp):
  """convert hex center coordinate hp to cartesian centerpoint cp"""
  i,j = hp
  cp = np.array([
    i*(2*R) + j*R,
    j*(S+T)
  ])
  cp = np.multiply(cp, SCALE)

return cp