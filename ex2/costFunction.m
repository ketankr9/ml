function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
J2=0; %duplicate
grad2=zeros(size(theta)); %duplicate
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% ========================iteration=====================
temp=0;
for i=1:m,
h=sigmoid(X'(:,i)'*theta);
temp=temp+(y(i)*log(h)   +    (1-y(i))*log(1-h));
%temp=temp + log( ((1-h)*(h/(1-h))^y(i)) );
end;
J2=-temp/m;
temp2=0;
for i=1:size(theta),
grad2(i)=0;
for j=1:m,
h=sigmoid(X'(:,j)'*theta);
temp2=temp2+(h-y(j)) * X(j,i);
end;
grad2(i)=temp2/m;
end;
% ==========================vectorization===================================
h=sigmoid(X*theta);
J=sum(-y.*log(h)-(1-y).*log(1-h))/m;

grad=sum(X.*(h-y))'/m;



% ===============================================================
end
