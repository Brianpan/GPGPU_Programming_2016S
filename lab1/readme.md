# Question 1 解題思路

a. 建立segment tree, offset逐步跑方式 起始node idx為1

b. bottom up trace 

	1. 先把起始點作到第二層方便作業
	
	2. 從每個點trace parent，終止條件loop_iv > 512(最大長度)或fail 
		. 是奇數不管
		
		. 是偶數看左邊的sibling是否大於等於loop_iv否就fail，是增加length  

c. top down

	1. 開始的點是從上一部最後的idx-1開始，中間考量到跨512高度的tree
	
	2. 往下trace看左右child，終止條件trace到最底層 leaf
		. 先看right child 是否是0是0往下trace

		. 不是0加上該right child的值並trace left child

*注意思考：

最高到512所以要考慮跨tree的問題怎麼解決，(sol: 向左挪一位)

考慮到最左邊的512高度tree，位置是奇數,且值不等於0 (sol: 不需往上找,也不需down trace)     
