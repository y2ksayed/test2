#Lindsey Bang
#Project1

#Homework1
def palindrome(n):
    x=''
    n=str(n)
    for i in range(len(n),0,-1):
        x=x+n[i-1]
    if x == n:
        return True
    else:
        return False
print()

def max_palindrome():
    list=[]
    firstmaxlist=[]
    secondmaxlist=[]
    max=""    
    for x1 in range(100,1000):
        for x2 in range(100,1000):
            x3 = x1*x2
            if palindrome(x3) == True and x3 not in list:
                list.append(x3)
                firstmaxlist.append(x1)
                secondmaxlist.append(x2)
    max=(sorted(list))[-1]
    onemax=(sorted(firstmaxlist))[-1]
    twomax=(sorted(secondmaxlist))[-1]
    return print(str(max)," (",str(onemax),", ",str(twomax),")")

max_palindrome()


#homework2
def sum_prime_number(n):
    prime=[2]
    for i in range(3,n):
        for x in prime:
            if i%x ==0:
                break
            if x ==max(prime):
                prime.append(i)
    return prime
print(sum(sum_prime_number(2000)))
#homework3

def sum_multiples(x,y,z):
    list = []
    for i in range(1,z):
        if i%x == 0 or i%z == 0:
            list.append(i)
    return print(sum(list))
sum_multiples(3,5,1000)

#homework4
def string_compressor(string):
    list=""
    count=1
    for i in range(1,len(string)):
        if string[i] == string[i-1]:
            count +=1
        else:
            list += (string[i-1]+str(count))
            count = 1
    list += (string[i]+str(count)) 
    return print(list)
string_compressor("aabcccccaaa")

#homework5-bonus challenge
def fizzbuzz(x,y,z,n):
    list=[]
    for i in range(1,n+1):
        if i%z==0:
            list.append("FizzBuzz")
        elif i%y==0:
            list.append("Fizz")
        elif i%x == 0: 
            list.append("Buzz")
        else:
            list.append(i)
    return print(list)
fizzbuzz(3,5,15,100)
