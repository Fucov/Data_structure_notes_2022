[TOC]



#### 1.小技巧

##### 1、重复元素判定

以下方法可以检查给定列表是不是存在重复元素，它会使用 set() 函数来移除所有重复元素。

```python
def all_unique(lst):
return len(lst)== len(set(lst))
x = [1,1,2,2,3,2,3,4,5,6]
y = [1,2,3,4,5]
all_unique(x) # False
all_unique(y) # True

```

##### 2、字符元素组成判定

检查两个字符串的组成元素是不是一样的。

```python
from collections import Counter
def anagram(first, second):
return Counter(first) == Counter(second)
anagram("abcd3", "3acdb") # True

```

##### 3、内存占用

```python
import sys
variable = 30
print(sys.getsizeof(variable)) # 24

```



##### 4、字节占用

下面的代码块可以检查字符串占用的字节数。

```python
def byte_size(string):
return(len(string.encode('utf-8')))
byte_size('') # 4
byte_size('Hello World') # 11
```



##### 5、打印 N 次字符串

该代码块不需要循环语句就能打印 N 次字符串。

```python
n = 2
s ="Programming"
print(s * n)

# ProgrammingProgramming
```



##### 6、大写第一个字母

以下代码块会使用 title() 方法，从而大写字符串中每一个单词的首字母。

~~~python


```
s = "programming is awesome"
print(s.title())
```



# Programming Is Awesome

~~~



##### 7、分块

给定具体的大小，定义一个函数以按照这个大小切割列表。

```python
from math import ceil
def chunk(lst, size):
	return list(map(lambda x: lst[x * size:x * size + size],list(range(0, ceil(len(lst) / size)))))
chunk([1,2,3,4,5],2)

# [[1,2],[3,4],5]
```



##### 8、压缩

这个方法可以将布尔型的值去掉，例如（False，None，0，“”），它使用 filter() 函数。

```python
def compact(lst):
	return list(filter(bool, lst))
compact([0, 1, False, 2, '', 3, 'a', 's', 34])
 [ 1, 2, 3, 'a', 's', 34 ]
```



##### 9、解包

如下代码段可以将打包好的成对列表解开成两组不同的元组。

```python
array = [['a', 'b'], ['c', 'd'], ['e', 'f']]
transposed = zip(*array)
print(transposed)

# [('a', 'c', 'e'), ('b', 'd', 'f')]


```



##### 10、链式对比

我们可以在一行代码中使用不同的运算符对比多个不同的元素。

```python
a = 3
print( 2 < a < 8) # True
print(1 == a < 2) # False

```



##### 11、逗号连接

下面的代码可以将列表连接成单个字符串，且每一个元素间的分隔方式设置为了逗号。

```python
hobbies = ["basketball", "football", "swimming"]
print("My hobbies are: " + ", ".join(hobbies))

# My hobbies are: basketball, football, swimming
```



##### 12、元音统计

以下方法将统计字符串中的元音 (‘a’, ‘e’, ‘i’, ‘o’, ‘u’) 的个数，它是通过正则表达式做的。

```python
import re
def count_vowels(str):
return len(len(re.findall(r'[aeiou]', str, re.IGNORECASE)))
count_vowels('foobar') # 3
count_vowels('gym') # 0

```



##### 13、首字母小写

如下方法将令给定字符串的第一个字符统一为小写。

```python
def decapitalize(string):
	return str[:1].lower() + str[1:]
decapitalize('FooBar') # 'fooBar'
decapitalize('FooBar') # 'fooBar'

```



##### 14、展开列表

该方法将通过递归的方式将列表的嵌套展开为单个列表。

```python
def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
        	ret.extend(i)
        else:
        	ret.append(i)
    return ret
def deep_flatten(lst):
    result = []
    result.extend(
    spread(list(map(lambda x: deep_flatten(x) if type(x) == list else x, lst))))
    return result
deep_flatten([1, [2], [[3], 4], 5]) # [1,2,3,4,5]
```



##### 15、列表的差

该方法将返回第一个列表的元素，其不在第二个列表内。如果同时要反馈第二个列表独有的元素，还需要加一句 set_b.difference(set_a)。

```python
def difference(a, b):
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)
    return list(comparison)
difference([1,2,3], [1,2,4]) # [3]

```

##### 16、通过函数取差

如下方法首先会应用一个给定的函数，然后再返回应用函数后结果有差别的列表元素。

```python
def difference_by(a, b, fn):
    b = set(map(fn, b))
    return [item for item in a if fn(item) not in b]
from math import floor
difference_by([2.1, 1.2], [2.3, 3.4],floor) # [1.2]
difference_by([{ 'x': 2 }, { 'x': 1 }], [{ 'x': 1 }], lambda v : v['x'])

# [ { x: 2 } ]
```



##### 17、链式函数调用

你可以在一行代码内调用多个函数。

```python
def add(a, b):
	return a + b
def subtract(a, b):
	return a - b
a, b = 4, 5
print((subtract if a > b else add)(a, b)) # 9
```



##### 18、检查重复项

如下代码将检查两个列表是不是有重复项。

```python
def has_duplicates(lst):
	return len(lst) != len(set(lst))
x = [1,2,3,4,5,5]
y = [1,2,3,4,5]
has_duplicates(x) # True
has_duplicates(y) # False

```



##### 19、合并两个字典

下面的方法将用于合并两个字典。

```python
def merge_two_dicts(a, b):
    c = a.copy() # make a copy of a 
    c.update(b) # modify keys and values of a with the once from b
    return c
a={'x':1,'y':2}
b={'y':3,'z':4}
print(merge_two_dicts(a,b))
#{'y':3,'x':1,'z':4}

在 Python 3.5 或更高版本中，我们也可以用以下方式合并字典：

def merge_dictionaries(a, b)
	return {**a, **b}
a = { 'x': 1, 'y': 2}
b = { 'y': 3, 'z': 4}
print(merge_dictionaries(a, b))
 {'y': 3, 'x': 1, 'z': 4}
```



##### 20、将两个列表转化为字典

如下方法将会把两个列表转化为单个字典。

```python
def to_dictionary(keys, values): return dict(zip(keys, values)) keys =
	[“a”, “b”, “c”] values = [2, 3, 4] print(to_dictionary(keys, values))
#{‘a’: 2, ‘c’: 4, ‘b’: 3}
```



##### 21、使用枚举

我们常用 For 循环来遍历某个列表，同样我们也能枚举列表的索引与值。

```python
list = ["a", "b", "c", "d"]
for index, element in enumerate(list): 
print("Value", element, "Index ", index, )

# ('Value', 'a', 'Index ', 0)

# ('Value', 'b', 'Index ', 1)

#('Value', 'c', 'Index ', 2)

# ('Value', 'd', 'Index ', 3)
```



##### 22、执行时间

如下代码块可以用来计算执行特定代码所花费的时间。

```python
import time
start_time = time.time()
a = 1
b = 2
c = a + b
print(c) #3
end_time = time.time()
total_time = end_time - start_time
print("Time: ", total_time)

# ('Time: ', 1.1205673217773438e-05) 

```



##### 23、Try else

我们在使用 try/except 语句的时候也可以加一个 else 子句，如果没有触发错误的话，这个子句就会被运行。

```python
try:
	2*3
except TypeError:
	print("An exception was raised")
else:
	print("Thank God, no exceptions were raised.")
#Thank God, no exceptions were raised.
```



##### 24、元素频率

下面的方法会根据元素频率取列表中最常见的元素。

```python
def most_frequent(list):
	return max(set(list), key = list.count)
list = [1,2,1,2,3,2,1,4,2]
most_frequent(list)

```



##### 25、回文序列

以下方法会检查给定的字符串是不是回文序列，它首先会把所有字母转化为小写，并移除非英文字母符号。最后，它会对比字符串与反向字符串是否相等，相等则表示为回文序列。

```python
def palindrome(string):
    from re import sub
    s = sub('[\W_]', '', string.lower())
    return s == s[::-1]
palindrome('taco cat') # True

```



##### 26、不使用 if-else 的计算子

这一段代码可以不使用条件语句就实现加减乘除、求幂操作，它通过字典这一数据结构实现：

```python
import operator
action = {
"+": operator.add,
"-": operator.sub,
"/": operator.truediv,
"*": operator.mul,
"**": pow
}
print(action['-'](50, 25)) # 25
```



##### 27、Shuffle

该算法会打乱列表元素的顺序，它主要会通过 Fisher-Yates 算法对新列表进行排序：

```python
from copy import deepcopy
from random import randint
    def shuffle(lst):
        temp_lst = deepcopy(lst)
        m = len(temp_lst)
        while (m):
            m -= 1
            i = randint(0, m)
            temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
        return temp_lst
foo = [1,2,3]
shuffle(foo) # [2,3,1] , foo = [1,2,3]
```



##### 28、展开列表

将列表内的所有元素，包括子列表，都展开成一个列表。

```python
def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
        	ret.extend(i)
        else:
       		ret.append(i)
    return ret
spread([1,2,3,[4,5,6],[7],8,9]) # [1,2,3,4,5,6,7,8,9]
```



##### 29、交换值

不需要额外的操作就能交换两个变量的值。

```python
def swap(a, b):
	return b, a
a, b = -1, 14
swap(a, b) # (14, -1)
spread([1,2,3,[4,5,6],[7],8,9]) # [1,2,3,4,5,6,7,8,9]

```

##### 30、字典默认值

通过 Key 取对应的 Value 值，可以通过以下方式设置默认值。如果 get() 方法没有设置默认值，那么如果遇到不存在的 Key，则会返回 None。

```python
d = {'a': 1, 'b': 2}
print(d.get('c', 3)) # 3
```

#### 2.动态规划常见问题

##### 1、**最少硬币数**

有3种硬币，其面值为2元、5元、7元，现在要拼成27元，并要求硬币数量最少

备注：每种硬币数量都无穷多

```python
#一
coin=sorted(list(map(int,input().split())))
m=int(input())
sumlist=[-1]*max((m+1),coin[-1]+1)
for i in coin:
    sumlist[i]=1
for i in coin:  # 计算每一个最小,n为当前要计算的
    for n in range(i,m+1):
        if sumlist[n-i]==-1:
            continue
        elif sumlist[n]==-1:
            sumlist[n]=sumlist[n-i]+1
        else:
            sumlist[n]=min(sumlist[n],sumlist[n-i]+1)
print(sumlist[n])
#二
money=int(input())
num=sorted(list(map(int,input().split())))
x=[0 for i in range(money+1)]#到达当前钱数，各个硬币用一次，选最少的一个
for i in range(1,money+1):
    y=[float("inf")]*len(num)
    for j in range(len(num)):
        if i-num[j]>=0:
            y[j]=x[i-num[j]]+1
    x[i]=min(y)
print(x[-1])
```

##### **2、不同路径**

给定m行n列的网格，有一个机器人从网格[0,0]处出发，且只能往右或往下走，有多少条路径可以走到右下角？

```python
def countWays(x, y):
    dp = [[1]*y for i in range(x)]
    for i in range(1,x):
        for j in range(1,y):
            dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
    return dp[x - 1][y - 1]
n,m=map(int,input().split())
print(countWays(n,m))
```

##### 3、jump game

给定一个非负整数[数组](https://so.csdn.net/so/search?q=数组&spm=1001.2101.3001.7020)，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个位置。

```python
def canJump( nums):
    L = len(nums)
    if L == 0:
        return False
    G = nums[0]
    for i in range(1, L):
        if G < i:
            return False
        G = max(G, nums[i] + i)
        if G >= L - 1:
            return True
    return G >= L - 1
lst=list(map(int, input().split()))
print(canJump(lst))
```

如果能跳到，最少需要几步？

```python
def jump( nums):
    curReach,lastReach,cnt,Len=0,0,0,len(nums)
    for i in range(0,Len):
        if lastReach < i :
            lastReach = curReach
            cnt += 1
        curReach = max(curReach,nums[i]+i)
    return cnt
lst=list(map(int, input().split()))
print(jump(lst))
```

##### 4、分割等和子集

给定一个**只包含正整数**的**非空**数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```python
def canPartition( nums):
    n = len(nums)
    C = sum(nums) // 2
    if n == 0:
        return True
    if sum(nums) % 2:  #全体和是奇数
        return False
    memo = [False for i in range(C + 1)]
    for i in range(0, C + 1):
        memo[i] = (nums[0] == i)
    for i in range(1, n):
        for j in range(C, nums[i] - 1, -1):
            memo[j] = memo[j] or memo[j - nums[i]]
    return memo[C]
lst=list(map(int, input().split()))
print(canPartition(lst))
```

##### 5、买卖股票

```python
#1.给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
def maxProfit(prices):
    if (len(prices) == 0):
        return 0;
    maxprofit, minprice = [0] * len(prices), prices[0]

    for i in range(1, len(prices)):
        maxprofit[i] = max(maxprofit[i - 1], prices[i] - minprice)
        minprice = min(prices[i], minprice)

    return maxprofit[-1]
prices=list(map(int,input().split()))
print(maxProfit(prices))

#给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。 对于交易次数没有限制注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
def maxProfit( prices) :
    maxProfit = 0
    for i in range(1, len(prices)):
        if (prices[i - 1] < prices[i]):
            maxProfit = maxProfit + prices[i] - prices[i - 1]
    return maxProfit
prices=list(map(int,input().split()))
print(maxProfit(prices))

#给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
def maxProfit(prices):
    if len(prices) < 1:
        return 0
    if len(prices) // 2 <= 2:
        maxProfit = 0
        for i in range(1, len(prices)):
            if (prices[i - 1] < prices[i]):
                maxProfit += prices[i] - prices[i - 1]
        return maxProfit

    dp = [0] * (3)
    v = [prices[0]] * (3)
    for i in range(1, len(prices)):
        for t in range(1, 3):
            v[t] = min(v[t], prices[i] - dp[t - 1])
            dp[t] = max(dp[t], prices[i] - v[t])

    return dp[2]
prices=list(map(int,input().split()))
print(maxProfit(prices))

#给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格.设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
def maxProfit( k, prices):
    if len(prices) < 1:
        return 0
    if len(prices) // 2 <= k:
        maxProfit = 0
        for i in range(1, len(prices)):
            if (prices[i - 1] < prices[i]):
                maxProfit += prices[i] - prices[i - 1]
        return maxProfit

    dp = [0] * (k + 1)
    v = [prices[0]] * (k + 1)
    for i in range(1, len(prices)):
        for t in range(1, k + 1):
            v[t] = min(v[t], prices[i] - dp[t - 1])
            dp[t] = max(dp[t], prices[i] - v[t])

    return dp[k]
prices=list(map(int,input().split()))
k=int(input())
print(maxProfit(k,prices))

```

