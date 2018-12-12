HOW TO:
1. How to draw robotic arms:
    type in:
        python3 SimBase.py

2. How to generate data:
    type in:
        python3 gendata.py
    
    Edit savepath(line 20 in gendata.py) to set save path of data
    Edit VideoNum(line 22 in gendata.py) to set number of data


    The generated data is stored in .npz format. It can be regarded as a dictionary:
    {
        "video":a np.array of size (2,64,64,3) which stores 2 images,
        "cubeindx": the index of the moved cube,
        "actionindx": the index of the action type
    }

3. How to change the image size:

    Edit rendersetting["screenwidth"] and rendersetting["screenwidth"] in gendata.py(line 27,28) to set the height and width of generated data

4. TroubleShoot: when type in 
        python3 gendata.py
    the error can occur:
        RuntimeError: Failed to initialize OpenGL

    Please type in:
        unset LD_PRELOAD
    and try again
