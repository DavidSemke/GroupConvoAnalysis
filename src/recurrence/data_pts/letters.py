from src.utils.token import is_word

def letter_data_pts(convo):
    data_pts = []
    letters = [chr(i) for i in range(97, 123)]
    letter_to_index = {p[0]:p[1] for p in zip(letters, range(26))}

    for utt in convo.iter_utterances():
        words = utt.text.split()

        for word in words:
            
            if not is_word({'tok': word}): continue
            
            word = word.lower()
            data_pts.extend(
                [letter_to_index[letter] for letter in word]
            )
        
    return data_pts