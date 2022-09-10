# 字节跳动企业题库

leetcode上的企业题库，这里记录字节跳动的题库

## 一、简单题出现频率top50



## 二、中等题出现频率top100

### 1、两数相加[**LinkedList当栈用**]

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/add-two-numbers
<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220909131413977.png" alt="image-20220909131413977" style="zoom: 33%;" />

如果当前两个链表处相应的数字是n1、n2，进位为carry，则其对应的和的数字为（n1+n2+carry）mod 10，新的进位值是（n1+n2+carry）mod 10向下取整

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
      // 将链表化成list比较好处理
      // LinkedList实现了栈和队列的操作方法 所以可以作为栈来使用
       LinkedList<Integer> list1=buildList(l1);
       LinkedList<Integer> list2=buildList(l2);
      
       List<ListNode> sumList=new ArrayList<>();
      
       int carry=0;
      
       // 在默写时这个条件可能不是很容易想出
       // 其逻辑是 只要进位或者任意一个数对应位置有值 即可继续做加法 
       while(!list1.isEmpty() || !list2.isEmpty()||carry!=0){
         // 当作栈来使用
         int x=list1.isEmpty()?0:list1.pop();
         int y=list2.isEmpty()?0:list2.pop();
         int sum=x+y+carry;
         // 生成node
         ListNode node=new ListNode(sum%10);
         sumList.add(node);
          
         // carry
         carry=sum/10;
         
       }
      
        // 重建list而已
       for(int i=0;i+1<sumList.size();i++){
            sumList.get(i).next=sumList.get(i+1);
        }

        return sumList.get(0);
       
      
      
    }
  
    private LinkedList<Integer> buildList(ListNode l){
      LinkedList<Integer> res=new LinkedList<>();
      if(l==null){
        return res;
      }
      
      while(l!=null){
        res.add(l.val);
        l=l.next;
      }
      return res;
    }
  
}
```

### 2、无重复字符的最长子串[滑动窗口]

https://leetcode.cn/problems/longest-substring-without-repeating-characters/

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220909140027483.png" alt="image-20220909140027483" style="zoom:33%;" />

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220909183111818.png" alt="image-20220909183111818" style="zoom:50%;" />

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s==null||s.length()==0){
            return 0;
        }
        char [] chars=s.toCharArray();
        int len=chars.length;
        int left=0;
        int right=0;
        int maxLength=0;
        Set<Character> set=new HashSet<>();
        // 所以1. 外层循环是左指针
        while(right<len){

            //  3. 不满足条件 指的是使得左指针右移的条件
            // 刚开始是不满足的
            while(set.contains(chars[right])){
                // 3.1 移除left对应的元素
                // 3.2 left++
                set.remove(chars[left]);
                left++;
            }

            // 4.满足条件
            set.add(chars[right]);
            maxLength= Math.max(maxLength, right - left + 1);
            right++;
        }

        return maxLength;

    }
}
```

### 3、长度最小的字数组[滑动窗口]

https://leetcode.cn/problems/minimum-size-subarray-sum/

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220909161419503.png" alt="image-20220909161419503" style="zoom:33%;" />

```java

class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int left=0;
        int right=0;
        int minLength=0;
        int currentSum=0;
        while(right<nums.length){
            currentSum+=nums[right];
            // 2. 如果满足条件 收集  则左指针向右边移动
            while(currentSum>=target){
                if(minLength>right-left+1 || minLength==0){
                    minLength=right-left+1;
                }
                //  3 向后移动
                currentSum-=nums[left];
                left++;
            }

            right++;

        }

        return minLength;
    }

}


```

### 4、最长回文子串[***]

### 4、最长回文子序列[***]

### 4、最长公共子序列[***]

### 5、LRU缓存[哈希表与双向链表]

https://leetcode.cn/problems/lru-cache/

请你设计并实现一个满足  `LRU` (最近最少使用) 缓存 约束的数据结构。

实现 `LRUCache` 类：
`LRUCache(int capacity)` 以 正整数 作为容量 `capacity` 初始化 `LRU` 缓存

`int get(int key)` 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。

`void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值` value` ；

如果不存在，则向缓存中插入该组 `key-value` 。

如果插入操作导致关键字数量超过 `capacity` ，则应该 逐出 最久未使用的关键字。

函数 `get` 和` put` 必须以 `O(1)` 的平均时间复杂度运行。

```java
class LRUCache {

    // 由于get和put都需要以o（1）的时间复杂度运行  所以可以使用hash表
    // put时  可能会删除节点 要o（1）使用双向链表
    // 使用虚拟头尾节点  可以统一处理插入

    // 双向链表
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;

        public DLinkedNode() {
        }
        public DLinkedNode(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }




    // 哈希表
    private Map<Integer,DLinkedNode> cache=new HashMap<>();
    // 双向链表的两个头尾节点
    private DLinkedNode head;
    private DLinkedNode tail;
    // 容量
    private int capacity;
    // 已经使用的空间
    private int size;

    // 初始化
    public LRUCache(int capacity) {
        this.size=0;
        this.capacity=capacity;
      // 初始化头部尾部节点
        head=new DLinkedNode();
        tail=new DLinkedNode();
        head.next=tail;
        tail.prev=head;
    }

    // get一次就需要将节点移动到头部
    public int get(int key) {
        DLinkedNode node = cache.get(key);
        // 如果没有在哈希表中找到 key  
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部  
        move2Head(node);
        return node.value;
    }

    // put方法要生成新数或者新节点 并将该节点移到头部
    public void put(int key, int value) {
        DLinkedNode node=cache.get(key);
        if (node==null){
            // 插入新的节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            // size++ 如果大于容量限制  则需要将尾部节点删除
            size++;
            if (size > capacity) {
                // 得到删除的节点
                DLinkedNode removed = removeTail();
                // 根据得到的 key 删除哈希表中的元素
                cache.remove(removed.key);
                // 减少已使用容量
                size--;
            }
            // 没有大于容量限制的直接插入hash表  并添加node即可
            // 插入哈希表
            cache.put(key, newNode);
            // 添加至双链表的头部
            add2Head(newNode);
        }else{
            // 变更节点的值
            node.value=value;
            // 移动到头部  所以该方法最好做的是删除节点  并移动该节点到头部新增
            move2Head(node);
        }


    }


    // 添加
    private void add2Head(DLinkedNode node){
        // 可以想象一下加到头部的场景
        // 新节点到旧的节点的连接先建立
        node.prev=head;
        node.next=head.next;

        // 旧节点到新节点的连接建立 从后往前
        head.next.prev=node;
        head.next=node;
    }

    // 删除
    private void removeNode(DLinkedNode node){
        // 新建node左右两个邻居节点的连接而已
        node.prev.next=node.next;
        node.next.prev=node.prev;
    }


    // 移动旧的到头 即删除并添加到头
    private void move2Head(DLinkedNode node){
        removeNode(node);
        add2Head(node);
    }

    // 移除尾部节点，淘汰最久未使用的
    private DLinkedNode removeTail(){
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;

    }
}

```

### 6、三数之和[双指针]

leetcode 15 

> 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
>
> 注意：答案中不可以包含重复的三元组。

思路：双指针

如何三个数使用双指针，肯定是遍历数组，==挑出一个数，其他两个数互为相反数==，`其他两个数的遍历使用双指针`

<font color=red>在这之前</font>，要明白，双指针由于有快慢指针之分，或者方向指针之分，其中==方向指针一般适用于有序数组==

<font color=red>这里，我们就需要使用方向指针</font>,即一个left指针，一个right指针。一般是在<font color=red>有序数组</font>上使用。

所以我们`需要对数组排序`

解释以下代码的三个逻辑

1.`nums[left]+nums[right]==target`时，自然`res.add(Arrays.asList(nums[i],nums[left],nums[right]));`

且在这个逻辑下，必须跳过`++left与left对应数组值相等的情况`,同理必须跳过`--right与right对应值相等的情况`

2.在`nums[left]+nums[right]`小于`target`时左移，则`left++`，相反则`right--`

3.很容易忘记的一点是，每次`遍历的第一个数如果和前一个数相同`，应该需要`跳过该数的逻辑以以去重`

```java
class Solution {
        public List<List<Integer>> threeSum(int[] nums) {
            List<List<Integer>> res=new ArrayList<>();
            if(null==nums||nums.length<3){
                return res;
            }

            // 1、最基础的一步 先排序
            int len=nums.length;
            Arrays.sort(nums);

            // 双指针
            for(int i=0;i<len;i++){
                // 去重第一步
                if (i>0 && nums[i]==nums[i-1])
                    continue;
                int target=-nums[i];
                // left开始是从i的下一个开始
                int left=i+1;
                int right=len-1;
                while(right>left) {
                    if (nums[left] + nums[right] == target) {
                        res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                        // 这里跳过去重一定要有 以免重复
                        while (left < right && nums[left] == nums[++left]) ;
                        while (left < right && nums[right] == nums[--right]) ;
                    }

                    if (nums[left] + nums[right] > target) {
                        right--;
                    }

                    if (nums[left] + nums[right] < target) {
                        left++;
                    }

                }

            }
            
            return res;

        }
    }
```

### 7、盛最多水的容器[双指针]

https://leetcode.cn/problems/container-with-most-water/

双指针，其实也就是哪边比较小哪边移动

```java
class Solution {
   public int maxArea(int[] height) {

        int len = height.length;
        int left=0;
        int right=len-1;
        int max=0;
        while(left<right){
            int tmp=(right-left)*Math.min(height[left],height[right]);
            max=Math.max(max,tmp);
            if(height[left]<height[right]){
                left++;
            }else{
                right--;
            }

        }


        return max;

    }
}
```

### 8、下一排列[***]

### 9、最大子数组和[坐标型动态规划]

https://leetcode.cn/problems/maximum-subarray/

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

设f[i]为结尾的连续子数组的最大和

这是个坐标型动态规划

```java
class Solution {
    public int maxSubArray(int[] nums) {

        // 肯定用动态规划
        // 最困难的是确定状态：我们应该以f[i]表示以对应位置的数结尾的子数组的最大值

        int len =nums.length;
        if(len==1){
            return nums[0];
        }

        int [] f=new int [len];
        f[0]=nums[0];
        int max=f[0];
        for(int i=1;i<len;i++){
            if(f[i-1]>0){
                f[i]=f[i-1]+nums[i];
            }else{
                f[i]=nums[i];
            }
            max=max>f[i]?max:f[i];
        }

        return max;
    }
}
```

### 10、合并区间[扫描线]

https://leetcode.cn/problems/merge-intervals/

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220910202745504.png" alt="image-20220910202745504" style="zoom:33%;" />

```java
class Solution {
     public int[][] merge(int[][] intervals) {
            // 这种不会不知道吧
            Arrays.sort(intervals, (o1, o2) -> o1[0]-o2[0]);
            List<List<Integer>> res=new ArrayList<>();
            int len=intervals.length;
            int start=intervals[0][0];
            int end=intervals[0][1];
            boolean isOverlapp=false;
            for (int i=1;i<len;i++){
                int[] interval = intervals[i];
                // [1,3] [2,6]
                // 相交 或者 包含
                if (interval[0]<=end ){
                    end=Math.max(end,interval[1]);
                    // 表示当前是重叠的
                    isOverlapp=true;
                }else{
                    // 隔离
                    // 收集非重叠区间
                    // 并重置start end
                    res.add(Arrays.asList(start,end));
                    // 表示当前非重叠
                    isOverlapp=false;
                    start=interval[0];
                    end=interval[1];
                }
            }

            // 因为出现下一个才会处理以上的  所以最后还是要处理最后一个
            if (!isOverlapp){
                res.add(Arrays.asList(intervals[len-1][0],intervals[len-1][1]));
            }else{
                res.add(Arrays.asList(start,end));
            }

            int [][] result=new int[res.size()][2];
            for (int i=0;i<res.size();i++){
                result[i][0]=res.get(i).get(0);
                result[i][1]=res.get(i).get(1);
            }


            return result;
        }
}
```

### 11、搜索旋转排序数组[二分法]

https://leetcode.cn/problems/search-in-rotated-sorted-array/

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220910205016707.png" alt="image-20220910205016707" style="zoom:33%;" />

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220910204944957.png" alt="image-20220910204944957" style="zoom:33%;" />

```java
class Solution {
    public int search(int[] nums, int target) {
        int left=0,right=nums.length;
        // 在左闭右开区间时right=数组的长
        // 1、寻找拐点
        // [left,right)
        while(left<right){
            // 中间点用这个表达式吧 不去加1
            int mid=(right+left)/2;
            if (nums[mid]>=nums[0]){
                // 表示还在递增   拐点在右边
                // 所以left+1 左闭 拐点一定在mid+1之后 含mid+1
                left=mid+1;
            }else{
                // 小于的话right=mid 因为右开
                right=mid;
            }
        }

        // 2、 确定区间
        if (target>=nums[0]){
            // 在左区间
            left=0;
        }else{
           
            right=nums.length;
        }

        // 3. 查找
        while(left<right){
            // 中间点用这个表达式吧 不去加1
            int mid=(right+left)/2;
            if (nums[mid]>target){
                // 表示在左边  把右指针移过来
                right=mid;
            }else if(nums[mid]<target){
                // 表示在右边  把左指针移过来
                left=mid+1;
            }else{
                return mid;
            }
        }

        return -1;

    }

}
```

总结二分法的两个模板

```java
// [left,right)的情况
//  初始化
int left=0, right=size;
while（left<right）{
  int mid=(left+right)/2;
  if (nums[mid]==target){
    ...
  }
  if (nums[mid]<target){
    // 说明在右边  left需要右移动  因为mid已经不可能了  所以要+1 而且left是闭的
    left=mid+1；
  }
  
  if (nums[mid]>target){
    // 说明在左边 right需要左移动  因为mid已经不可能了  所以right=mid 因为right是开的
    right=mid；
  }
}


// [left,right]的情况
//  初始化
int left=0, right=size-1;
while（left<=right）{
  int mid=(left+right)/2;
  if (nums[mid]==target){
    ...
  }
  if (nums[mid]<target){
    // 说明在右边  left需要右移动  因为mid已经不可能了  所以要+1 而且left是闭的
    left=mid+1；
  }
  
  if (nums[mid]>target){
    // 说明在左边 right需要左移动  因为mid已经不可能了  所以right=mid-1 因为right是闭的
    right=mid-1；
  }
}
 
// 感觉第一种比较好
```

### 12、重排链表[***]

### 13、数组中的第k个最大元素[最小堆]

https://leetcode.cn/problems/kth-largest-element-in-an-array

给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220910231934311.png" alt="image-20220910231934311" style="zoom:33%;" />

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
       // 优先队列  最小堆
        PriorityQueue<Integer> heap=new PriorityQueue<>();
        
        // 如果遍历的比他大 则替换最小的
       for(int num: nums){
           if (heap.size()<k)
               heap.add(num);
           else if (heap.peek()<num){
               heap.poll();
               heap.add(num);
           }
       }
       
       return heap.poll();
    }
}
```

始终拿到最小堆的元素去比较

### 14、岛屿数量[***]

### 15、整数反转[按位转换、数学]

https://leetcode.cn/problems/reverse-integer/

给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−2^31,  2^31 − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。

```java
class Solution {
   public int reverse(int x) {

        // res计作反转后的数字
        int res=0;
        // 12345
        // 5    1234

        // 重复弹出x的末尾数字
        while(x!=0){
            int digit=x%10;
            // 这里得先把异常给过滤掉 因为后面有乘以10的操作
            // 要么res*10大于  要么res*10+末尾>7 因为Integer.MAX_VALUE=2147483647
            if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && digit > 7))
                return 0;
            if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && digit < -8))
                return 0;
            // 余数越靠前得出  乘以的10越多 所以有以下
            res=res*10+digit;
            x/=10;

        }

        return res;

    }
}
```

### 16、二叉树的右视图[BFS]

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220911030533294.png" alt="image-20220911030533294" style="zoom:33%;" />

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    private List<List<Integer>>resList=new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {

        List<Integer> res=new ArrayList<>();
        bfs(root);
        
        resList.forEach(it->{
            res.add(it.get(it.size()-1));
        });

        return res;
    }

    //BFS--迭代方式--借助队列
    public void bfs(TreeNode node) {
        if(node==null){
            return;
        }
        Queue<TreeNode> que=new LinkedList<>();
        // 1. 首先把node加入队列中
        que.offer(node);

        while(!que.isEmpty()){
            List<Integer> ls=new ArrayList<>();
            int len =que.size();
            while (len>0){
                TreeNode tmpNode=que.poll();
                ls.add(tmpNode.val);
                
                 // 收集左右子树  先加入进去
                if (tmpNode.left != null) que.offer(tmpNode.left);
                if (tmpNode.right != null) que.offer(tmpNode.right);
                len--;

            }

            // 收集了一行
            resList.add(ls);
            
        }

    }
}
```

### 17、括号生成[***回溯]

### 18、复原ip地址[***回溯]

### 19、螺旋矩阵[方向数组]

https://leetcode.cn/problems/spiral-matrix/

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {



        List<Integer> res=new ArrayList<>();
        int m=matrix.length;
        int n=matrix[0].length;
        boolean [][] visited=new boolean[m][n];

        int [] dx={0,1,0,-1};
        int [] dy={1,0,-1,0};

        int x=0,y=0,dir=0;
        for(int i=0;i<m*n; i++){
            res.add(matrix[x][y]);
            visited[x][y]=true;

            // 先判断一下是否合法
            int  xtmp=x+dx[dir], ytmp=y+dy[dir];
            if(xtmp<0 ||xtmp>=m || ytmp<0 || ytmp>=n ||visited[xtmp][ytmp]){
                // 确定方向
                dir=(dir+1)%4;
            }

            x+=dx[dir];
            y+=dy[dir];

        }

        return res;

    }
}
```

### 19、螺旋矩阵II[方向数组]

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

<img src="/Users/dexlace/private-github-repository/leetcode-coding/byte-dance.assets/image-20220911041050154.png" alt="image-20220911041050154" style="zoom:33%;" />



## 三、困难题出现频率top50



