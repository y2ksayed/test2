def is_palindrome(n):#define function to see if number is palindrome
    n_as_string = str(n)#make number string
    
    n_reverse_string = n_as_string[::-1]#reverse string
    
    n_reverse_num = int(n_reverse_string)#make reverse string integer again
    
    if n_reverse_num == n:#compare reverse to original for truth
        
        return True
    
    else:
        
        return False
  
products_of_3_digit_nums = []#initiate list
for i in range(100,1000):#ake loop for three digit numbers
    for j in range(100,1000):#make nested loop to multiply all three digit numbers with each other
        products_of_3_digit_nums.append(i*j)#append to list
        #note it's not important if there are duplicates in list because we just want maximum
palindrome_nums = []#initiate list        
for num in products_of_3_digit_nums:#evaluate each product if true
    if is_palindrome(num) == True:
        palindrome_nums.append(num)#append to list
        
print(max(palindrome_nums))#find max