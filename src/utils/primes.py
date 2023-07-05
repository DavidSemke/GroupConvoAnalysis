def generate_primes(count):
    sequence = [2]
    num = 3

    while len(sequence) < count:
        
        for i in range(2, num):
            if num % i == 0: break
        
        else:
            sequence.append(num)
        
        num += 1
    
    return sequence