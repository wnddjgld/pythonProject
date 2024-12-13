from pasta.base.formatting import append


def myfunction(a, b):
    return a * b, a/b
re, le = myfunction(3, 2)


def fun():
    global a
    a=5
    print(a)
a=100
print(a)
fun()
# 지역변수를 지정안해주면 전역변수 사용함 print(a)

# print(re)
# print(le)

if __name__ == "__main__":
    pass


myList = []
for i in range(6):
    # myList.append(i)
    myList += [i]
print(myList)

myiist = [i for i in range(6)]
print(myiist)


MYLLIST = []
i=0
while i < 6:
    MYLLIST.append(i)
    i += 1
print(MYLLIST)