"""
Pseudocode:
Gives an outline of what the code will look like
class TreeNode:
    self.left_node
    self.right_node
    self.left_prediction
    self.right_prediction

    def predict_one(x):
        if self.condition(x):
            if self.left_node:
                return self.left_node.predict_one(x)
            else:
                return self.left_prediction
        else:
            if self.right_node:
                return self.right_node.predict_one(x)
            else:
                self.left_prediction

    def fit(X, Y):
        Best_IG = 0
        Best_attribute = None
        for c in columns:
            Condition = find_split(X, Y, c) # later explained
            Y_left = Y[X[c] meets condition]
            Y_right = Y[X[c] does not meet condition]
            Information_gain = H(Y) - p(left)H(Y_left) - p(right)H(Y_right)
            if Information_gain > Best_IG:
                Best_IG = Information_gain
                Best_attribute = c


#now must call fit recursively
X_left, Y_left, X_right, Y_right = split by best Best_attribute 
self.left_node = TreeNode()
self.left_node.fit(X_left, Y_left)
self.right_node = TreeNode()
self.right_node.fit(X_right, Y_right)
"""