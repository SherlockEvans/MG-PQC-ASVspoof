import numpy as np
import time
import datetime

def KeyGen(lam, q, n, eta):
    """
    :param q: 模数（一个大素数）
    :param n: 维度参数
    :param sigma: 高斯分布的标准差
    :return: 公钥 (pk)，私钥 (sk)
    """
    
    # 从均匀分布Z_q^(n x n)中生成矩阵A，数据类型为int64
    A = np.random.randint(low=0, high=q, size=(n, n), dtype=np.int64)
    
    # 从高斯分布D_(Z_n, sigma)中生成秘密向量s
    # s = np.random.normal(loc=0, scale=sigma, size=(n,))
    s = np.random.binomial(eta, 0.5, size=(n,))
    
    # 从高斯分布D_(Z_n, sigma)中生成误差向量e
    # e = np.random.normal(loc=0, scale=sigma, size=(n,))
    e = np.random.binomial(eta, 0.5, size=(n,))
    
    # 计算 b = A * s + e
    b = (np.dot(A, s) + e) % q
    
    # 公钥 (A, b)，私钥 s
    pk = (A, b)
    sk = s
    
    return pk, sk

def Enc(pk, mu, p, q, eta):
    """
    加密算法
    :param pk: 公钥 (A, b)
    :param mu: 明文向量 (mu 属于 Z_p^n)
    :param p: 模数 p
    :param q: 模数 q
    :param sigma: 高斯分布的标准差
    :return: 密文 (ct0, ct1)
    """
    A, b = pk
    
    # 从高斯分布 D_(Z_n×n, sigma) 中生成矩阵 X 和 E
    # X = np.random.normal(loc=0, scale=sigma, size=A.shape)
    # E = np.random.normal(loc=0, scale=sigma, size=A.shape)
    X = np.random.binomial(eta, 0.5, size=A.shape)
    E = np.random.binomial(eta, 0.5, size=A.shape)
    
    # 从高斯分布 D_(Z_n, sigma) 中生成向量 f
    f = np.random.binomial(eta, 0.5, size=b.shape)
    
    # 计算密文 (ct0, ct1)
    XA = np.dot(X, A) + E
    Xb = np.dot(X, b)
    ct0 = XA % q # XA = Xb - Xe
    ct1 = (Xb + f + (q // p) * mu) % q
    
    return (ct0, ct1)


def Dec(sk, ct, p, q):
    """
    解密算法
    :param sk: 私钥 s
    :param ct: 密文 (ct0, ct1)
    :param p: 模数 p
    :param q: 模数 q
    :return: 解密后的明文 mu
    """
    s = sk
    ct0, ct1 = ct
    
    # 计算 v = ct1 - ct0 * s
    v = (ct1 - np.dot(ct0, s)) % q
    
    # 返回解密后的明文 mu
    mu = np.round(p / q * v) % p
    return mu

def UpdatePk(pk, eta):
    """
    公钥更新算法
    :param pk: 当前公钥 (A, b)
    :param sigma: 高斯分布的标准差
    :return: 更新后的公钥 (pk') 和更新参数 (up)
    """
    A, b = pk
    
    # 从高斯分布 D_(Z_n, sigma) 中生成向量 r 和 η
    r = np.random.binomial(eta, 0.5, size=b.shape)
    eta = np.random.binomial(eta, 0.5, size=b.shape)
    
    # 计算更新后的公钥 (pk') 和更新参数 (up)
    b_prime = (b + np.dot(A, r) + eta) % q
    up = Enc(pk, r, p, q, eta)  # 使用 Enc 加密 r，生成更新参数 up
    
    return (A, b_prime), up

def UpdateSk(sk, up, p, q):
    """
    私钥更新算法
    :param sk: 当前私钥 s
    :param up: 更新参数 up
    :param p: 模数 p
    :param q: 模数 q
    :return: 更新后的私钥 sk'
    """
    # 使用解密算法 Dec(sk, up) 解密更新参数 up 得到 r
    r = Dec(sk, up, p, q)
    
    # 计算更新后的私钥 sk'
    sk_prime = sk + r
    return sk_prime

lam = 128  # 安全参数
q = 2**21  # 一个大的素数模数
p = 5  # 另一个较小的素数模数
n = 768  # 维度参数,正常是3*256
eta = 2  # 二项分布尝试次数
u = 100 # 密钥更新次数

# 生成密钥
public_key, private_key = KeyGen(lam, q, n, eta)

# 明文
mu = np.random.randint(low=0, high=p, size=(n,))
# mu = np.random.uniform(low=0, high=p, size=(n,))
print("明文:", list(mu))

# 加密
ciphertext = Enc(public_key, mu, p, q, eta)
# print("密文:", ciphertext)

# 解密
decrypted_mu = Dec(private_key, ciphertext, p, q)
de_mu = [int(x) if isinstance(x, float) else x for x in list(decrypted_mu)]
print("解密后的明文:", list(de_mu))

if list(mu) == list(de_mu):
    print("验证解密成功")
else:
    print("验证解密失败")
    

start_time = time.time()
current_time = datetime.datetime.now()

print("时间戳:", start_time)
print("当前时间:", current_time)    
for i in range(0,u):
    print('更新%d次密钥后, 验证结果....' % (i+1))

    # 更新公钥
    if i==0:
        new_public_key, update_param = UpdatePk(public_key, eta)
    else:
        new_public_key, update_param = UpdatePk(new_public_key, eta)
    
    # 更新私钥
    if i==0:
        new_private_key = UpdateSk(private_key, update_param, p, q)
    else:
        new_private_key = UpdateSk(new_private_key, update_param, p, q)
        
    # 明文
    mu = np.random.randint(low=0, high=p, size=(n,))
    # mu = np.random.uniform(low=0, high=p, size=(n,))
    # print("明文:", list(mu))

    # 更新后加密
    ciphertext = Enc(new_public_key, mu, p, q, eta)
    # print("密文:", ciphertext)

    # 更新后解密
    decrypted_mu = Dec(new_private_key, ciphertext, p, q)
    de_mu = [int(x) if isinstance(x, float) else x for x in list(decrypted_mu)]
    # print("解密后的明文:", list(de_mu))

    if list(mu) == list(de_mu):
        print("验证解密成功!")
    else:
        print("验证解密失败")

end_time = time.time()
execution_time = end_time - start_time
print("程序运行时间（秒）:", execution_time)
