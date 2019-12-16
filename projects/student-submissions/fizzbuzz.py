for i in range(1,101):
    status = ''#initiate empty string
    if i % 3 == 0:
        status = status + 'Fizz'#concatenate fizz if divisibe by 3
    if i % 5 == 0:#concatenate buzz if divisible by 5
        status = status + 'Buzz'
    if status == '':
        print(i)#print number if nothing concatenated
    else:
        print(status)#if something was concatenated, print string