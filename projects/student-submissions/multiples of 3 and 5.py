list_of_3_multiples = []
for i in range(1000):
    if i % 3 == 0:
        list_of_3_multiples.append(i)#append all mulltiples of 3
        
list_of_5_multiples = []
for j in range(1000):
    if j % 5 == 0:
        list_of_5_multiples.append(j)#append all multiples of 5
        
all_multiples = list_of_3_multiples + list_of_5_multiples#combine lists of multiples

deduped_all_multiples = list(set(all_multiples))#use set to deduplicate list

print(sum(deduped_all_multiples))#print sum of list

