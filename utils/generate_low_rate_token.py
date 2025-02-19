OUTPUT_DIR = '../model/codet5/none/token_attentions'

def generate_low_rated_token():
    with open(OUTPUT_DIR + '/token_attentions', 'r') as f:
        with open(OUTPUT_DIR + '/low_rated_word', 'w') as f2:
            for i in range(0, 5000):
                a = f.readline()
                f2.write(a.split('  ||  ')[0] + '\n')



if __name__ == '__main__':
    generate_low_rated_token()