require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("tools/SDAECriterionGPU.lua")


----------------------------------
-- Configuration
----------------------------------

local inputSize = 11916

local hiddenLayers = {100, 50 , 100}  -- Hidden layers' size

local sgdConfig = {
   learningRate      = 0.1, 
   learningRateDecay = 0.2,
   weightDecay       = 0.025,   -- lambda: L2-norm of W
   momentum          = 0.8,     -- Momentum
}
-- local adamConfig = {
--   learningRate       = 0.001,
--   -- learningRateDecay  = 0.2,
--   weightDecay        = 0.025,
--   beta1              = 0.9,
--   beta2              = 0.999,
-- }

local lossFct = cfn.SDAECriterionGPU(nn.MSECriterion(), {
   alpha     = 1.0,
   beta      = 1.0,
   hideRatio = 0.1,
}, inputSize)
lossFct.sizeAverage = false


local batchSize = 35
local epoches = 10



----------------------------------
-- Loading data
----------------------------------
print("Start Loading data...")

-- Step 1 : Load file
local trainfile = io.open("./data/yahoo_movie/train.csv", "r")
local testfile = io.open("./data/yahoo_movie/test.csv", "r")


local train, test = {}, {}  -- Trainset and testset
local itemHash = {}         -- Item hasher (from 1 to n)
local itemCnt  = 1          -- Item counter

local function parseLine(line)
  -- parse the file by using regex
  local userIdStr, movieIdStr, ratingStr = line:match('(%d+),(%d+),(%d+)')

  local userId  = tonumber(userIdStr)
  local itemId  = tonumber(movieIdStr)
  local rating  = tonumber(ratingStr)

  -- Hashing item id.
  itemIdx = itemHash[itemId]
  if itemIdx == nil then
    itemHash[itemId] = itemCnt
    itemIdx = itemCnt
    itemCnt = itemCnt + 1
  end

  -- normalize the rating between [-1, 1]
  rating = (rating-3)/2

  return userId, itemIdx, rating
end

-- Step 2 : Retrieve ratings
trainfile:read()  -- skip first line of train.csv
for line in trainfile:lines() do
   local userId, itemId, rating = parseLine(line)
   train[userId] = train[userId] or nnsparse.DynamicSparseTensor()
   train[userId]:append(torch.Tensor{itemId, rating})
end

testfile:read()  -- skip first line of test.csv
for line in testfile:lines() do
  local userId, itemId, rating = parseLine(line)
  test[userId] = test[userId] or nnsparse.DynamicSparseTensor()
  test[userId]:append(torch.Tensor{itemId, rating})
end


-- Step 3 : Build the final sparse matrices
for k, oneTrain in pairs(train) do train[k] = oneTrain:build():ssortByIndex() end
for k, oneTest  in pairs(test) do test[k]  = oneTest:build():ssortByIndex() end


-- Step 4 : remove mean
for k, oneTrain in pairs(train) do 
  local mean = oneTrain[{ {},2 }]:mean()
  train[k][{ {},2 }]:add(-mean) 
   
  if test[k] then 
    test[k] [{ {},2 }]:add(-mean) 
  end
end



----------------------------------
-- Building the network
----------------------------------
print("Start Building the network...")

local network = nn.Sequential()
-- Input layer
network:add(nnsparse.SparseLinearBatch(inputSize, hiddenLayers[1]))
network:add(nn.Tanh())
-- Hidden layers
for k = 2, #hiddenLayers do
  network:add(nn.Linear(hiddenLayers[k-1], hiddenLayers[k]))
  network:add(nn.Tanh())
end
-- Decode layer
network:add(nn.Linear(hiddenLayers[#hiddenLayers], inputSize))
network:add(nn.Tanh())

print(network)


----------------------------------
-- Training the network
----------------------------------

local function trainNN(network, t)

   -- Create minibatch
   local input, minibatch = {}, {}

   --shuffle the indices of the inputs to create the minibatch 
   local shuffle = torch.randperm(inputSize)
   shuffle:apply(function(k)
      if train[k] then
         input[#input+1] = train[k] 
         if #input == batchSize then
            minibatch[#minibatch+1] = input
            input = {}  
         end
      end
   end)
   if #input > 0 then 
      minibatch[#minibatch+1] = input 
   end


   local w, dw = network:getParameters()
   lossFct.sizeAverage = false

   -- Classic training 
   for _, input in pairs(minibatch) do
      local function feval(x)

         -- Reset gradients and losses
         network:zeroGradParameters()

         -- AutoEncoder targets
         local target = input

         -- Compute noisy input for Denoising autoencoders
         local noisyInput = lossFct:prepareInput(input) 

         -- FORWARD
         local output = network:forward(noisyInput)
         local loss   = lossFct:forward(output, target)

         -- BACKWARD
         local dloss = lossFct:backward(output, target)
         local _     = network:backward(noisyInput, dloss)

         -- Return loss and gradients
         return loss/batchSize, dw:div(batchSize)
      end
      
      sgdConfig.evalCounter = t
      optim.sgd (feval, w, sgdConfig)
      -- optim.adam (feval, w, adamConfig)
   end
end





----------------------------------
-- Testing the network
----------------------------------

local function testNN(network)

   local criterion = nnsparse.SparseCriterion(nn.MSECriterion())

   -- Create minibatch
   local noRatings = 0
   local input, target, minibatch = {}, {}, {}
   
   for k, _ in pairs(train) do
   
     if test[k] ~= nil then --ignore when there is no target 
       input [#input  +1] = train[k] 
       target[#target +1] = test[k]
       
       noRatings = noRatings + test[k]:size(1)
     
       if #input == batchSize then
         minibatch[#minibatch+1] = {input = input, target = target}
         input, target = {}, {}  
       end
     end
   end
   if #input > 0 then 
      minibatch[#minibatch+1] = {input = input, target = target} 
   end

   -- define the testing criterion
   local criterion = nnsparse.SparseCriterion(nn.MSECriterion())
   criterion.sizeAverage = false

   -- Compute the RMSE by predicting the testing dataset thanks to the training dataset
   local err = 0
   for _, oneBatch in pairs(minibatch) do
     local output = network:forward(oneBatch.input)
     err = err + criterion:forward(output, oneBatch.target)
   end
   
   err = err/noRatings

   print("Current RMSE : " .. math.sqrt(err) * 2)

end
   
print("Start Training the network...")   
for t = 1, epoches do
   xlua.progress(t, epoches)
   print('')
   trainNN(network, t)
   testNN(network)
end
   

   