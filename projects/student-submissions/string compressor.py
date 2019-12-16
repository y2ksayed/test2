def string_compress(s):
    s = s.lower()#lower case string, function not case sensitive
    list_of_lists = []#initiate list to hold list of same letters
    start_break_index = 0#start index at 0 to know where to begin string slice
    for i in range(len(s) - 1):# loop through strings using indices
        if s[i] != s[i+1]:# if current string index not the same as next one, then proceed
            end_break_index = i#i marks where to end slice
            all_same_letters = s[start_break_index:end_break_index + 1]#all_same_letters is created
            list_of_lists.append(all_same_letters)#append all_same_letters to list of lists
            start_break_index = end_break_index + 1#start break index now is end break plus 1
    
    reconstruct = ''#loop above will omit last group of same letters from list of lists
    #we must add in the last group of same letter to list of lists
    for letters in list_of_lists:#reconstruct string minus the last letters from original string
        reconstruct = reconstruct + letters
        
    num_letters_missing = len(s) - len(reconstruct)#find how many letters are missing
    
    letters_missing = s[-num_letters_missing:]#obtain missing letters from original string
    
    list_of_lists.append(letters_missing)#append the missing letters to list of lists
    
    letter = []#initiate lists to gather the letter of num of times it appears
    num_times = []
    for item in list_of_lists:
        letter.append(item[0])#obtain first letter in each list
        num_times.append(len(item))#append how many times that letter appears
        
    compressed = ''    #make blank string
    for let, num in zip(letter, num_times):#loop through the linked lists
        compressed = compressed + str(let) + str(num)#concatenate into string letter and number of appearance
        
    if len(s) <= len(compressed):#return orignal string if shorter or same then compressed string
        return s
    else:
        return compressed
        
        
        