# Coding

A list of algorithms commonly used to solve leetcode problems

## Two Pointers

Time Complexity: O(N)
- Array or linked list
- Limit focus to range of elements from input to consider **subset** of data
- No exhaustive search, where eliminating one solution does not eliminate others
  
```python
start, end = 0, len(array) - 1

while left < right:
    # do some logic here with left and right
    if CONDITION:
        left += 1
    else:
        right -= 1

return ans
```

## Fast and Slow Pointers

Time Complexity: O(N)/O(logn) 
- Identify first x% of elements in list
- Element at the k-way point in list eg. middle element, second quartile
- Cycle detection

```python
slow, fast = head, head

while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

    # do something here ...
```

## Sliding Window

Time Complexity: O(n)
- Repeated computations on contiguous set of elements

```python
def fn(arr):
    left = ans = curr = 0

    for right in range(len(arr)):
        # do logic here to add arr[right] to curr

        while WINDOW_CONDITION_BROKEN:
            # remove arr[left] from curr
            left += 1

        # update ans
    
    return ans
```

## Merge Intervals

Time Complexity: O(n)
- **Sorted** array of intervals as input
- Find intersection, union or gaps in intervals

```python

result = []
result.append(intervals[0])
for interval in intervals[1:]:
    prev_end = result[-1][1]
    curr_start, curr_end = interval

    if curr_start <= prev_end:
        result[-1][1] = max(curr_end, prev_end)
    else:
        result.append(interval)
```

## Merge Two Lists

Time Complexity: O(n)
- two sorted lists

```python
def merge_lists(l1, l2):
  head = curr = ListNode(0)

  while l1 and l2:
    if l1.val <= l2.val:
      curr.next = l1
      l1 = l1.next
    else:
      curr.next = l2
      l2 = l1
      l1 = curr.next.next

  if not l1:
    curr.next = l2
  else:
    curr.next = l1

  return head.next
  
```

## Quickselect
Time Complexity: O(n) average, O(n^2) worst case

> By chosing a random pivot we increase the chances of equal sized partitions,
> Drastically different sized partitions causes the recursion depth to become large.

- Top K elements

```python
def topk(arr, K): 

    def partition(low, high):
        # chose random pivot, move pivot to end of array
        # want random pivot to lessen the chances
        pivot = random.randint(low, high)
        arr[high], arr[pivot] = arr[pivot], arr[high]

        pivot = arr[high]
        i = low - 1

        # move elements that are smaller than the pivot to 
        # the left of the pivot
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]

        return i + 1

    def quickselect(low, high, k):
        if low <= high:
            pivot = partition(low, high)

            if pivot == k: # found k-th smallest element
                return pivot
            elif pivot < k: # pivot less than k, search left
                return quickselect(pivot + 1, high, k)
            else:           # pivot greater than k, search right
                return quickselect(low, pivot - 1, k)

    
    quickselect(0, len(arr) - 1, K - 1)

    return arr[:K]
```

## Backtracking
Time Complexity: Difficult to compute. 
> Example is O(SUM[n,k=1]{P(N,k)} where P(N,k) = N!/(N-k)! -> N(N-1) ... (N-k + 1)
> 
> Simplified its better than O(NxN!) and slower than O(N!)
- Useful for combinations or permutations problems
- Build solution along the traversal path

```python
def permute(array):
  result = []

  def backtracking(current_values, permutation):
    # meets target case
    if len(permutation) == len(array):
      result.append(permutation)

    # if invalid case, return. Stop exploring
    lf len(permutation) > len(array):
      return

    for i, value in enumerate(current_values):
      # skip current value in list of possible values. Add that value to permutation
      backtracking(current_values[:i] + current_values[i+1:], permutation + [value])

  backtrack(array, [])

  return result
```
    
## Topological Sort
Time Complexity: Linear usually O(edges + vertices)
- Graph problem, find partial ordering based on dependency rules
- Can detect if graph is a dag or not

```python
def toposort(dependencies):
  num_dependencies = len(dependencies)

  # 1. populate graph
  in_degrees = [ 0 for _ in range(num_dependencies) ]
  adjacency_list = [ [] for _ in range(num_dependencies) ]

  for child, parent in dependencies:
    adjacency_list[parent].append(child)
    in_degrees[child] += 1

  # 2. initialize queue with root nodes (in degree is 0)
  q = deque( [ node for node, degree in enumerate(in_degrees) if degree == 0] )

  # 3. BFS over adjacency list
  order = []
  while q:
    node = q.popleft()
    order.append(node) 

    for child in adjacency_list[node]:
      # remove child from graph. all new root nodes get added to order.
      in_degree[child] -= 1
      if in_degree[child] == 0:
        q.append(child)

  # 4. DAG check. A valid solution will have all vertices in order
  if len(order) != num_dependencies:
    return []

  return order
```

## Trie
Time Complexity: Depends on operation. But usually linear O(len_words) dominated by creating the tree.
- Space optimized word storage
- Prefix matching

```python
ENDCHAR = '$'

def add_word(trie, word):
  current = trie

  for char in word:
    if char not in current:
      current[char] = {}
    current = current[char]

  # could just be None too to save space
  current[ENDCHAR] = word

def search(trie, word, is_prefix=False):
  current = trie

  for char in word:
    if char not in current:
      return False
    current = current[char]
  
  return is_prefix or ENDCHAR in current

def create_trie(words):
  trie = {}

  for word in words:
    add_word(trie, word)

  return trie
```

## Breadth First Serach
Time Complexity: O(n)
- explore graph
- level order traversal of tree

```python
def bfs(head):

  visited = set()
  q = deque([(1, head)])

  while q:
    level, node = q.popleft()
    
    # visit node here
    visited.add(node)

    for child in node.children:
      if child and child not in visited:
        q.append((level + 1, child))     
```

## Depth First Serach
Time Complexity: O(n)
- explore graph
- in-order traversal of Tree

### Recursive (easiest)
```python
       1
      / \
     2   3
    / \
   4   5

# 1 2 4 5 3 
def preorder(node):
  if not node: return

  visited.add(node)
  preorder(node.left)
  preorder(node.right)

# 4 2 5 1 3 
def inorder(node):
  if not node: return

  inorder(node.left)
  visited.add(node)
  inorder(node.right)

# 4 5 2 3 1 
def postorder(node):
  if not node: return

  postorder(node.left)
  postorder(node.right)
  visited.add(node)
```
### Iterative
```python
def inorder_interative(root):
  stack = []
  node = root

  while node or stack:
    # traverse left
    while node:
      stack.append(node)
      node = node.left

    # visit current node
    node = stack.pop()
    visited.append(node)

    node = node.right

def preorder_iterative(root):
    stack = [root]
    visited = []

    while stack:
        node = stack.pop()
        visited.append(node)

        # push the right child first so its processed after left child.
        for next in [node.right, node.left]
          if next:
            stack.append(next)

    return visited

def postorder_iterative(root):
    stack = [root]
    visited = []

    while stack:
        node = stack.pop()
        visited.append(node)

        # push the left child first so its processed after right child.
        for next in [node.left, node.right]
          if next:
            stack.append(next)

    # result in reverse order for post-order traversal
    return [node for node in reversed(visited)]
```

## Binary Search
Time Complexity: O(logn)
- find value in sorted array or array that has segmets of sorted values (like if the sorted array was rotated around a pivot)

```python
def binary_search(nums, target):
    low = 0
    high = len(nums) - 1
    
    while low <= high:
        mid = low + ((high - low) // 2)
        if nums[mid] == target:
            return mid
        elif target < nums[mid]:
            high = mid - 1
        elif target > nums[mid]:
            low = mid + 1

    return -1
```




































