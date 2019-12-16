#Define a function that returns true or false to decide is number is prime
def is_prime(n):
    #initiate list of divisors
    list_divisors = []
    for i in range(1,n+1):#check all number from 1 to n inclusive
        if n % i == 0:#check is any number in between is divisible
            list_divisors.append(i)#append number to list

    if list_divisors == [1,n]:
        #if divisors are just 1 and n, then prime
        return True
    
    else:
        #if there are other divisors, then not prime
        return False

sum_of_primes = 0 #Initiate counter
for i in range(2,2001):# check if 2 to 2000 inclusive are prime
    if is_prime(i) == True:#if i is prime, this will return true
        sum_of_primes = sum_of_primes + i# add to counter
        

print(sum_of_primes)










