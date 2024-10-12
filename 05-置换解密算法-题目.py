# 置换解密Decrypt算法 transposition cipher
 
cipherText = 'twsadr n tryngtI a  akadsom ih'
print('密文：', cipherText)

# 密文的一半长度
halfLength = len(cipherText)//2
# 前半部分为偶数位置字符串
evenChars  = cipherText[halfLength:]
# 后半部分为奇数位置字符串
oddChars  = cipherText [:halfLength]
# 将偶数、奇数位置字符串重新拼接成明文
plainText  = ''
for i in range(len(oddChars)):
    plainText += evenChars[i]
    plainText += oddChars[i]

if len(oddChars) < len(evenChars):
    # 补上最后一个偶数位置字符
    plainText += evenChars[ - 1 ]

print('明文：', plainText)

##输出：
##密文： twsadr n tryngtI a  akadsom ih
##明文： It was a dark and stormy night
