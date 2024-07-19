# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:17:42 2024

@author: harmuor
"""
a = 1
b = 2
c = 3
c and a
d = False
e = False

for i in range(1, 5, 2):
    if i//2 == 0:
        print("打死古优", i)
    else:
        print("干活古优", i)
        
        
i = 5
while i != 0:
    if i %2 ==0:
        print(i)
    i = i -1


i = 5
while i != 0:
    if i %2 ==0:
        print(i)
    i = i -1
    
for i in range(0, 10):
  if i == 1:
    print("执行了continue")
    continue   #跳过该次循环的代码，继续下一次循环
  print(i)
print("循环语句")

a = 1
while a < 10:
  a = a + 2
  if a == 5:
    continue
  print(a)
  
for i in range(5):
  if i == 2:
    break  #跳出全部循环
  print(i)
  
for i in range(2):
  for j in range(3):
      if j == 1:
          continue
      print(i, j)
  print("结束第", i+1, "次循环")
  
for i in range(2):
    for j in range(2, 4):
        print(i, j)
    print(i, j)
    

def myfun(a, b):
    c = a + b
    return c

myfun(24, 23)
print(c)

