# https://ko.linux-console.net/?p=6713 참조
'''
파이썬에서 문자열끼리 대소 비교를 할 때에는 각 첫 글자부터 유니코드를 기준으로 대소문자를 비교한다
'''
print('30' < '100') # False
'''
파이썬에서 같은 글자끼리는 더 긴쪽이 큰쪽이 된다
'''
print('aaa' < 'aa' )# False   
print('aaa' > 'aa' )# True
'''
유니코드 기준이기에 대소문자에 유의한다
'''
print('a'<'B')  # False