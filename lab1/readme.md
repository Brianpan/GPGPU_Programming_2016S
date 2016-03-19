# Question 1 思路
1. 建立segment tree
2. kernel 在從每個點往上trace
	a. trace parent 奇數繼續往上
	b. 偶數的話判斷旁邊的node是否大於loop_iv, 否�fail
	c. top down trace 先看right child是否是0,是就從right繼續trace, 否就trace left child, 終止條件是到最底層

