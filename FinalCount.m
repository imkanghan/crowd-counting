function result = FinalCount(A)

[row, column] = size(A);
C = A;
C(2:2:(row - 1), :) = 0;
C(:, 2 : 2 : (column - 1)) = 0;
if mod(row, 2) == 0
   C(row, :) = C(row, :) / 2;  
end

if mod(column, 2) == 0
    C(:, column) = C(:, column) / 2;  
end

result = sum(sum(C));

end
