// ********************************
// Tensor Operations in the Browser
// ********************************

// NOTE: View results by opening up Chrome's developer tools
// and selecting the "console" tab. 

// Link to TFJS documentation: https://js.tensorflow.org/api/1.0.0/ 


//    ***
// 1. *** Tensor Construction ***
//    ***

// define a scalar


// define a 1d tensor


// define a 2d tensor


// if we try logging to the console in the usual way 
// we'll get the object, and not the values



// use the built-in print method instead



// convert a tensor back into an array




//    ***
// 2. *** Simple Math Operations ***
//    ***






//    ***
// 3. *** Memory Management ***
//    ***

// When running within the browser, we need to be
// conscientious of memory usage. If we are creating
// intermediate tensors are part of a computation, 
// we should dispose of them afterwards to remove
// their burden on memory. 



// Use tf.tidy() to automatically dispose of tensors that
// are no longer needed.

// construct x^3+ x^2 + x^1 + c
