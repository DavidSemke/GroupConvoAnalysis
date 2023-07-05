def generate_primes(count):
    
    if count == 0:
        raise Exception('Tried to generate 0 primes')
    
    sequence = [2]
    num = 3

    while len(sequence) < count:
        
        for i in sequence:
            if num % i == 0: break
        
        else:
            sequence.append(num)
        
        num += 1
    
    return sequence


def is_prime(num):

    if num < 2: return False

    for i in range(2, num):
        if num % i == 0: return False

    return True