import argparse

def arg_parse():
    parser=argparse.ArgumentParser(description='Testing')
    
    
    args=parser.parse_args()
    return args
    
    
if __name__ == '__main__':
    args = arg_parse()