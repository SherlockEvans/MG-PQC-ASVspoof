import unittest
import os
import json
from src.dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
import soundfile as sf
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import galois
import time
from concurrent.futures import ThreadPoolExecutor


class TestMLDSA(unittest.TestCase):
    """
    Test ML DSA for internal consistency by generating signatures
    and verifying them!
    """

    def generic_test_ml_dsa(self, ML_DSA, count=5):
        for _ in range(count):
            msg = b"Signed by ML_DSA" + os.urandom(16)
            ctx = os.urandom(128)

            # Perform signature process
            pk, sk = ML_DSA.keygen()
            sig = ML_DSA.sign(sk, msg, ctx=ctx)
            check_verify = ML_DSA.verify(pk, msg, sig, ctx=ctx)

            # Sign with external_mu instead
            external_mu = ML_DSA.prehash_external_mu(pk, msg, ctx=ctx)
            sig_external_mu = ML_DSA.sign_external_mu(sk, external_mu)
            check_external_mu = ML_DSA.verify(pk, msg, sig_external_mu, ctx=ctx)

            # Generate some fail cases
            pk_bad, _ = ML_DSA.keygen()
            check_wrong_pk = ML_DSA.verify(pk_bad, msg, sig, ctx=ctx)
            check_wrong_msg = ML_DSA.verify(pk, b"", sig, ctx=ctx)
            check_no_ctx = ML_DSA.verify(pk, msg, sig)

            # Check that signature works
            self.assertTrue(check_verify)

            # Check that external_mu also works
            self.assertTrue(check_external_mu)

            # Check changing the key breaks verify
            self.assertFalse(check_wrong_pk)

            # Check changing the message breaks verify
            self.assertFalse(check_wrong_msg)

            # Check removing the context breaks verify
            self.assertFalse(check_no_ctx)

    def test_ml_dsa_44(self):
        self.generic_test_ml_dsa(ML_DSA_44)

    def test_ml_dsa_65(self):
        self.generic_test_ml_dsa(ML_DSA_65)

    def test_ml_dsa_87(self):
        self.generic_test_ml_dsa(ML_DSA_87)


class TestMLDSADeterministic(unittest.TestCase):
    """
    Test ML DSA for internal consistency by generating signatures
    and verifying them!
    """

    def generic_test_ml_dsa(self, ML_DSA, count=5):
        for _ in range(count):
            msg = b"Signed by ML_DSA" + os.urandom(16)
            ctx = os.urandom(128)

            # Perform signature process
            pk, sk = ML_DSA.keygen()
            sig = ML_DSA.sign(sk, msg, ctx=ctx, deterministic=True)
            check_verify = ML_DSA.verify(pk, msg, sig, ctx=ctx)

            # Sign with external_mu instead
            external_mu = ML_DSA.prehash_external_mu(pk, msg, ctx=ctx)
            sig_external_mu = ML_DSA.sign_external_mu(
                sk, external_mu, deterministic=True
)
            check_external_mu = ML_DSA.verify(pk, msg, sig_external_mu, ctx=ctx)

            # Generate some fail cases
            pk_bad, _ = ML_DSA.keygen()
            check_wrong_pk = ML_DSA.verify(pk_bad, msg, sig, ctx=ctx)
            check_wrong_msg = ML_DSA.verify(pk, b"", sig, ctx=ctx)
            check_no_ctx = ML_DSA.verify(pk, msg, sig)

            # Check that signature works
            self.assertTrue(check_verify)

            # Check that external_mu also works
            self.assertTrue(check_external_mu)

            # Check changing the key breaks verify
            self.assertFalse(check_wrong_pk)

            # Check changing the message breaks verify
            self.assertFalse(check_wrong_msg)

            # Check removing the context breaks verify
            self.assertFalse(check_no_ctx)

    def test_ml_dsa_44(self):
        self.generic_test_ml_dsa(ML_DSA_44)

    def test_ml_dsa_65(self):
        self.generic_test_ml_dsa(ML_DSA_65)

    def test_ml_dsa_87(self):
        self.generic_test_ml_dsa(ML_DSA_87)

    def test_derive_with_wrong_seed_length(self):
        with self.assertRaises(ValueError) as e:
            ML_DSA_44.key_derive(bytes(range(31)))

        self.assertIn("seed must be 32 bytes long", str(e.exception))



# add new
def speech_pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def pqc_dsasign(msg):
    # msg_bytes = msg.tobytes()
    msg_bytes = msg
    ctx = os.urandom(128)
    pk, sk = ML_DSA_44.keygen()
    # print('ctx---:',ctx)
    # print('pk---:',pk)
    # print('sk---:',sk)
    # sig = ML_DSA_44.sign(sk, msg, ctx=ctx)
    # print('sig---:',sig)
    # check_verify = ML_DSA_44.verify(pk, msg, sig, ctx=ctx)
    # print(check_verify)
    # # Sign with external_mu instead
    external_msg = ML_DSA_44.prehash_external_mu(pk, msg_bytes, ctx=ctx)
    sig_external_msg = ML_DSA_44.sign_external_mu(sk, external_msg)
    # print('sig_external_msg---:', sig_external_msg)
    return pk, msg_bytes, sig_external_msg, ctx

def pqc_dsaverify(pk, msg, sig_external_msg, ctx):
    msg_bytes = msg
    check_external_mu1 = ML_DSA_44.verify(pk, msg_bytes, sig_external_msg, ctx=ctx)
    print('验签check_external_mu1:',check_external_mu1)



def tensorproduct_encry(a):
    gf = int(10000000000000061)
    GF = galois.GF(gf)  # 定义有限域
    # 1. 生成25x25随机矩阵
    p_mat = GF.Random((25, 25), low=0, high=10)  # 替代np.random.randint转换:ml-citation{ref="5" data="citationList"}
    b_mat = GF.Random((8, 8), low=0, high=10)
    # 2.
    # num_str = "12345678901234567890123456789012345678901234567890123456789012345678901234567"
    # front_str = num_str[:64]  # 取前64位
    # b = np.array([float(c) for c in front_str]).reshape(8, 8)
    # 3. 处理64600维向量
    a = GF(a)
    blocks = [a[i * 200:(i + 1) * 200] for i in range(323)]  # 分块
    # 计算Kronecker积
    kronecker_GF = np.kron(p_mat, b_mat)  # 形状应为(200, 200)

    # 正向运算
    forward_results = []
    for block in blocks:
        res = block @ kronecker_GF
        forward_results.append(res)
    forward_final = np.concatenate(forward_results)
    # def process_block(block):
    #     return block @ kronecker_GF
    # with ThreadPoolExecutor() as executor:
    #     forward_results = list(executor.map(process_block, blocks))
    # forward_final = np.concatenate(forward_results)
    return forward_final, p_mat, b_mat

def tensorproduct_decry(forward_final, p_mat, b_mat):
    # 逆向运算
    gf = int(10000000000000061)
    GF = galois.GF(gf)  # 定义有限域
    p_mat = GF(p_mat)
    b_mat = GF(b_mat)
    kronecker_GF = np.kron(p_mat, b_mat)  # 形状应为(200, 200)
    try:
        backward_blocks = np.split(forward_final, 323)
        kronecker_inv = np.linalg.inv(kronecker_GF)
        # identity_GF = GF.Identity(200)
        # kronecker_inv = GF(np.linalg.solve(kronecker_GF, identity_GF))
        # kx = kronecker_GF @ kronecker_inv
        # print(kx)
        backward_results = []
        for block in backward_blocks:
            res = block @ kronecker_inv
            backward_results.append(res)
        c = np.concatenate(backward_results)
        # print('c:', c)
        # 比较结果
        # print("原始a的形状:", a.shape, a.size, type(a))  #(64600,) 64600 <class 'galois.GF(
        # print("重建c的形状:", c.shape, c.size, type(c))   #(64600,) 64600 <class 'galois.GF(
        # print("严格相等:", np.array_equal(a, c))  #true
        return c
    except np.linalg.LinAlgError:
        print("矩阵不可逆，无法完成验证")


def max_decimal_places(arr):
    max_len = 0
    for num in arr.flatten():
        s = str(float(num))  # 避免科学计数法干扰
        if '.' in s:
            decimal_part = s.split('.')[1]
            max_len = max(max_len, len(decimal_part.rstrip('0')))  # 去除末尾无效0
    return max_len


def transform_array(arr):
    # 将数组乘以10^15并转为整数
    scaled = (arr * 1e15).astype(np.int64)
    # 对负数处理：第一位变为1
    negative_mask = scaled < 0
    transformed = np.abs(scaled)
    transformed[negative_mask] += 10 ** 15  # 将负数的第一位变为1
    return transformed


def restore_array(transformed):
    # 分离正负数
    negative_mask = transformed >= 10 ** 15
    restored = transformed.copy()
    # 还原负数
    restored[negative_mask] -= 10 ** 15
    restored[negative_mask] *= -1
    # 还原原始值
    return restored / 1e15


class AESManager:
    def __init__(self, key: bytes):
        """初始化 AES 加解密管理类，设置密钥"""
        self.key = key
        self.cipher = AES.new(self.key, AES.MODE_CBC)

    def encrypt(self, plaintext: str) -> bytes:
        """加密明文"""
        # data = plaintext.encode('utf-8')
        data = plaintext
        padded_data = pad(data, AES.block_size)  # 填充数据使其成为 128 位的倍数
        ciphertext = self.cipher.encrypt(padded_data)
        return self.cipher.iv + ciphertext  # 返回 IV 和密文

    def decrypt(self, ciphertext: bytes) -> str:
        """解密密文"""
        iv = ciphertext[:AES.block_size]  # 获取 IV
        ciphertext = ciphertext[AES.block_size:]  # 获取密文部分
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded_plaintext = cipher.decrypt(ciphertext)
        plaintext = unpad(padded_plaintext, AES.block_size)
        return plaintext


def concat_to_bytes(arr):
    # 将数组元素转为字符串并连接
    concat_str = ''.join(map(str, arr.flatten().tolist()))
    # 转换为bytes类型（默认UTF-8编码）
    return concat_str.encode('utf-8')

if __name__ == '__main__':
    x, fs = sf.read(r"E:\database\ASVspoof2019LA-Sim v1.0\train-part-part\LA_T_1001718.wav")
    # print(x)
    x_pad = speech_pad(x, 64600)  # print(x_pad.shape)   (64600,)  # print(x_pad.size)  64600  # print(type(x_pad))  <class 'numpy.ndarray'>
    # abd= max_decimal_places(x_pad)
    # print(abd)  15
    print('x_pad:', x_pad)
    x_pad_int = transform_array(x_pad)
    print('x_pad_int:', x_pad_int)
    x_pad_int_bytes = concat_to_bytes(x_pad_int)
    # print('x_pad_int_bytes',x_pad_int_bytes)

    start = time.time()
    # # dsa and encrypt
    pk, msg, sig_external_msg, ctx = pqc_dsasign(x_pad_int_bytes)
    encryptedvector, p_mat, b_mat = tensorproduct_encry(x_pad_int)
    #
    # #decrypt
    c_gf_int = tensorproduct_decry(encryptedvector, p_mat, b_mat)
    c_nparray_int = np.array(c_gf_int)
    c_int_bytes = concat_to_bytes(c_nparray_int)
    # # print(c_int_bytes)
    # # 验证签名
    pqc_dsaverify(pk, c_int_bytes, sig_external_msg, ctx)
    c = restore_array(c_nparray_int)
    end = time.time()
    print(end - start)
    print("转化字节类型同：", x_pad_int_bytes == c_int_bytes)
    print("c:", c)
    print("c与 x_pad:", c == x_pad)
    print(np.array_equal(x_pad, c))
    print(np.array_equal(x_pad_int, c_nparray_int))














