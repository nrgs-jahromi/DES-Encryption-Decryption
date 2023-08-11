import string

initial_perm = [58, 50, 42, 34, 26, 18, 10, 1,
                60, 52, 44, 36, 28, 20, 12, 4,
                62, 54, 46, 38, 30, 22, 14, 6,
                64, 56, 48, 40, 32, 24, 16, 8,
                57, 49, 41, 33, 25, 17, 9, 2,
                59, 51, 43, 35, 27, 19, 11, 3,
                61, 53, 45, 37, 29, 21, 13, 5,
                63, 55, 47, 39, 31, 23, 15, 7]

exp_d = [32, 1, 2, 3, 4, 5, 4, 5,
         6, 7, 8, 9, 8, 9, 10, 11,
         12, 13, 12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21, 20, 21,
         22, 23, 24, 25, 24, 25, 26, 27,
         28, 29, 28, 29, 30, 31, 32, 1]

sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

final_perm = [8, 40, 48, 16, 56, 24, 64, 32,
              39, 7, 47, 15, 55, 23, 63, 31,
              38, 6, 46, 14, 54, 22, 62, 30,
              37, 5, 45, 13, 53, 21, 61, 29,
              36, 4, 44, 12, 52, 20, 60, 28,
              35, 3, 43, 11, 51, 19, 59, 27,
              34, 2, 42, 10, 50, 18, 58, 26,
              33, 1, 41, 9, 49, 17, 57, 25]

keyp = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

key_comp = [14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32]

shift_table = [1, 1, 2, 2,
               2, 2, 2, 2,
               1, 2, 2, 2,
               2, 2, 2, 1]

# permut_dic=[]
key = "4355262724562343"

plain_and_cipher = [
    'kootahe,6E2F7B25307C3144',
    'Zendegi,CF646E7170632D45',
    'Edame,D070257820560746',
    'Dare,5574223505051150',
    'JolotYe,DB2E393F61586144',
    'Daame,D175257820560746',
    'DaemKe,D135603D1A705746',
    'Mioftan,D83C6F7321752A54',
    'Toosh,413A2B666D024747',
    'HattaMo,5974216034186B44',
    'khayeSa,EA29302D74463545',
    '05753jj,B1203330722B7A04',
    '==j95697,38693B6824232D231D1C0D0C4959590D',
]


# test_cipher = '59346E29456A723B62354B61756D44257871650320277C741D1C0D0C4959590D'

def pad_text(text):
    padding_length = 8 - (len(text) % 8)
    padding = chr(padding_length) * padding_length
    return text + padding


def convert_to_binary(text):
    binary = ''.join(format(ord(c), '08b') for c in text)
    return binary  # Truncate to 64 bits


def apply_permutation(arr, permutation):
    permuted_arr = [arr[i] for i in permutation]
    return permuted_arr


def subtract_one_from_array(arr):
    subtracted_arr = [num - 1 for num in arr]
    return subtracted_arr


def apply_permutation(arr, permutation):
    permuted_arr = [arr[i] for i in permutation]
    return permuted_arr


def divide_string(text):
    length = len(text)
    mid = length // 2
    first_half = text[:mid]
    second_half = text[mid:]
    return first_half, second_half


def xor_binary_strings(str1, str2):
    result = ""
    for bit1, bit2 in zip(str1, str2):
        xor_bit = str(int(bit1) ^ int(bit2))
        result += xor_bit
    return result


def divide_array(array):
    subarrays = []
    subarray_size = 6
    for i in range(0, len(array), subarray_size):
        subarray = array[i:i + subarray_size]
        subarrays.append(subarray)
    return subarrays


def concatenate_first_last_number(element):
    first_number = element[0]
    last_number = element[5]
    combined_value = first_number + last_number
    decimal_value = int(combined_value, 2)
    return decimal_value


def S_box(input_array):
    sbox_values = []
    decimal_valus = []
    i = 0
    for element in input_array:
        decimal_value = concatenate_first_last_number(element)
        sbox_value = bin(sbox[i][decimal_value][convert_bin_to_decimal(element[1:-1])])[2:]
        sbox_value = sbox_value.zfill(4)
        sbox_values.append(sbox_value)
        decimal_valus.append(decimal_value)
        i += 1

    return sbox_values


def convert_bin_to_decimal(bin_input):
    decimal_num = 0
    power = len(bin_input) - 1

    for digit in bin_input:
        decimal_num += int(digit) * 2 ** power
        power -= 1

    return decimal_num


def split_shift(key):
    str = ''.join(key)
    left_part = str[1:28] + str[0]
    right_part = str[29:] + str[28]
    return left_part + right_part


def find_permute(input, output):
    res = []

    for i in range(len(output)):
        matches = []

        index = -1

        while True:
            try:
                index = input.index(output[i], index + 1)
                matches.append(index)
            except ValueError:
                break

        res.append(matches)

    return res


def find_common_elements(list1, list2):
    common_element_resulte = []
    for i in range(len(list1)):
        common_elements = []
        for element in list1[i]:
            if element in list2[i]:
                common_elements.append(element)

        common_element_resulte.append(common_elements)

    return common_element_resulte


def f_function(cipher, key):
    expaned = apply_permutation(cipher, exp_d)
    xor_result = xor_binary_strings(key, expaned)
    sbox_input = divide_array(xor_result)
    sbox_output = S_box(sbox_input)
    sbox_out = ''.join(sbox_output)
    pbox = apply_permutation(sbox_out, straight_pbox)
    return pbox


def decrypt(cipher, key):
    cipher_bin = ''
    for char in cipher:
        cipher_bin += bin(int(char, 16))[2:].zfill(4)
    cipher_permuted = apply_permutation(cipher_bin, initial_perm)
    left_cipher, right_cipher = divide_string(cipher_permuted)
    f_output = f_function(right_cipher, key)
    xor_result = ''.join(xor_binary_strings(f_output, left_cipher))
    merged = xor_result + ''.join(right_cipher)
    final_permut = apply_permutation(merged, final_perm)
    plain = convert_to_ascii(''.join(final_permut))
    return plain


def convert_to_ascii(bin_str):
    new_str = ''
    for i in range(len(bin_str) // 8):
        new_str += chr(int(bin_str[i * 8:(i * 8) + 8], 2))
    return new_str


def isprintable(s):
    for char in s:
        if char not in string.printable:
            return False
    return True


if __name__ == "__main__":
    permut_result = []
    initial_perm = subtract_one_from_array(initial_perm)
    final_perm = subtract_one_from_array(final_perm)
    exp_d = subtract_one_from_array(exp_d)
    keyp = subtract_one_from_array(keyp)
    key_comp = subtract_one_from_array(key_comp)

    # ------

    binary_key = ""
    for char in key:
        binary_key += bin(int(char, 16))[2:].zfill(4)

    deleted_parity = apply_permutation(binary_key, keyp)
    shifted_key = split_shift(deleted_parity)
    round_key = apply_permutation(shifted_key, key_comp)

    # -----
    for item in plain_and_cipher:
        plain_text, cipher_text = item.split(",")
        padded = pad_text(plain_text)
        n = (len(padded) / 8)
        if n == 2:
            plain1, plain2 = divide_string(padded)
            cipher1, cipher2 = divide_string(cipher_text)
            new_pair1 = plain1 + ',' + cipher1
            new_pair2 = plain2 + ',' + cipher2

            # -------------------------------
            binary = convert_to_binary(plain2)
            after_permut = apply_permutation(binary, initial_perm)
            left_half, right_half = divide_string(after_permut)
            exp_output = apply_permutation(right_half, exp_d)
            xor_result = xor_binary_strings(round_key, exp_output)
            sbox_input = divide_array(xor_result)
            sbox_output = S_box(sbox_input)

            # ----------------------------------------

            binary_cihper = ""
            for char in cipher2:
                binary_cihper += bin(int(char, 16))[2:].zfill(4)

            permut_cipher = apply_permutation(binary_cihper, initial_perm)
            cipher_left_half, cipher_right_half = divide_string(permut_cipher)
            f_output = xor_binary_strings(cipher_left_half, left_half)
            result = find_permute(''.join(sbox_output), f_output)
            permut_result.append(result)

            # -------------------------------
            padded = plain1
            cipher_text = cipher1

        binary = convert_to_binary(padded)
        after_permut = apply_permutation(binary, initial_perm)
        left_half, right_half = divide_string(after_permut)
        exp_output = apply_permutation(right_half, exp_d)
        xor_result = xor_binary_strings(round_key, exp_output)
        sbox_input = divide_array(xor_result)
        sbox_output = S_box(sbox_input)

        # ----------------------------------------

        binary_cihper = ""
        for char in cipher_text:
            binary_cihper += bin(int(char, 16))[2:].zfill(4)

        permut_cipher = apply_permutation(binary_cihper, initial_perm)
        cipher_left_half, cipher_right_half = divide_string(permut_cipher)
        f_output = xor_binary_strings(cipher_left_half, left_half)
        result = find_permute(''.join(sbox_output), f_output)
        permut_result.append(result)

    first_exampl = permut_result[0]
    for item in permut_result:
        first_exampl = find_common_elements(first_exampl, item)

    straight_pbox = []

    for sublist in first_exampl:
        n = 0
        for i in range(len(sublist)):
            if sublist[i] in straight_pbox:
                continue
            else:
                straight_pbox.append(sublist[i])
                n = 0
                break

    # print("straight_pbox : ",straight_pbox)
    # ------------------------------------------------------------------------------------------------
    test_cipher = input()
    substrings = [test_cipher[i:i + 16] for i in range(0, len(test_cipher), 16)]
    divided_cipher = list(substrings)
    test_plain = []
    for i in range(len(divided_cipher)):
        divided_plain = decrypt(divided_cipher[i], round_key)
        test_plain.append(divided_plain)

    test_plain_text = ''
    for i in range(len(test_plain)):
        if isprintable(test_plain[i]):
            test_plain_text += test_plain[i]

    print(test_plain_text)
