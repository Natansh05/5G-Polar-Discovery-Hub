
#-----------------Group-3--------------------------------------------------#
#-----------------CT-216 Project-------------------------------------------#
#-----------------Polar Code-----------------------------------------------#
#-----------------Prof. Yash Vasavada--------------------------------------#
#-----------------Mentor: Radhika Agrawal (202211054)----------------------#


import numpy as np
import matplotlib.pyplot as plt
import  math as mt

R_Sequence= [1, 2, 3, 5, 9, 17, 33, 4, 6, 65, 10, 7, 18, 11, 19, 129, 13, 34, 
    66, 21, 257, 35, 25, 37, 8, 130, 67, 513, 12, 41, 69, 131, 20, 14,
    49, 15, 73, 258, 22, 133, 36, 259, 27, 514, 81, 38, 26, 23, 137,
    261, 265, 39, 515, 97, 68, 42, 145, 29, 70, 43, 517, 50, 75, 273,
    161, 521, 289, 529, 193, 545, 71, 45, 132, 82, 51, 74, 16, 321,
    134, 53, 24, 135, 385, 77, 138, 83, 57, 28, 98, 40, 260, 85, 139,
    146, 262, 30, 44, 99, 516, 89, 141, 31, 147, 72, 263, 266, 162,
    577, 46, 101, 641, 52, 149, 47, 76, 267, 274, 518, 105, 163, 54,
    194, 153, 78, 165, 769, 269, 275, 519, 55, 84, 58, 522, 113, 136,
    79, 290, 195, 86, 277, 523, 59, 169, 140, 100, 87, 61, 281, 90,
    291, 530, 525, 197, 142, 102, 148, 177, 143, 531, 322, 32, 201,
    91, 546, 293, 323, 533, 264, 150, 103, 106, 305, 297, 164, 93,
    48, 268, 386, 547, 325, 209, 387, 151, 154, 166, 107, 56, 329,
    537, 578, 549, 114, 155, 80, 270, 109, 579, 225, 167, 520, 553,
    196, 271, 642, 524, 276, 581, 292, 60, 170, 561, 115, 278, 157,
    88, 198, 117, 171, 62, 532, 526, 643, 282, 279, 527, 178, 294,
    389, 92, 585, 770, 199, 173, 121, 202, 337, 63, 283, 144, 104,
    179, 295, 94, 645, 203, 593, 324, 393, 298, 771, 108, 181, 152,
    210, 285, 649, 95, 205, 299, 401, 609, 353, 326, 534, 156, 211,
    306, 548, 301, 110, 185, 535, 538, 116, 168, 226, 327, 307, 773,
    158, 657, 330, 111, 118, 213, 172, 777, 331, 227, 550, 539, 388,
    309, 217, 417, 272, 280, 159, 338, 551, 673, 119, 333, 580, 541,
    390, 174, 122, 554, 200, 785, 180, 229, 339, 313, 705, 391, 175,
    555, 582, 394, 284, 123, 449, 354, 562, 204, 64, 341, 395, 528,
    583, 557, 182, 296, 286, 233, 125, 206, 183, 644, 563, 287, 586,
    300, 355, 212, 402, 186, 397, 345, 587, 646, 594, 536, 241, 207,
    96, 328, 565, 801, 403, 357, 308, 302, 418, 214, 569, 833, 589,
    187, 647, 405, 228, 897, 595, 419, 303, 650, 772, 361, 540, 112,
    332, 215, 310, 189, 450, 218, 409, 610, 597, 552, 651, 230, 160,
    421, 311, 542, 774, 611, 658, 334, 120, 601, 340, 219, 369, 653,
    231, 392, 314, 451, 543, 335, 234, 556, 775, 176, 124, 659, 613,
    342, 778, 221, 315, 425, 396, 674, 584, 356, 288, 184, 235, 126,
    558, 661, 617, 343, 317, 242, 779, 564, 346, 453, 398, 404, 208,
    675, 559, 786, 433, 358, 188, 237, 665, 625, 588, 781, 706, 127,
    243, 566, 399, 347, 457, 359, 406, 304, 570, 245, 596, 190, 567,
    677, 362, 707, 590, 216, 787, 648, 349, 420, 407, 465, 681, 802,
    363, 591, 410, 571, 789, 598, 573, 220, 312, 709, 599, 602, 652,
    422, 793, 803, 612, 603, 411, 232, 689, 654, 249, 370, 191, 365,
    655, 660, 336, 481, 316, 222, 371, 614, 423, 426, 452, 615, 544,
    236, 413, 344, 373, 776, 318, 223, 427, 454, 238, 560, 834, 805,
    713, 835, 662, 809, 780, 618, 605, 434, 721, 817, 837, 348, 898,
    244, 663, 455, 319, 676, 619, 899, 782, 377, 429, 666, 737, 568,
    841, 626, 239, 360, 458, 400, 788, 592, 679, 435, 678, 350, 246,
    459, 667, 621, 364, 128, 192, 783, 408, 437, 627, 572, 466, 682,
    247, 708, 351, 600, 669, 791, 461, 250, 683, 574, 412, 804, 790,
    710, 366, 441, 629, 690, 375, 424, 467, 794, 251, 372, 482, 575,
    414, 604, 367, 469, 656, 901, 806, 616, 685, 711, 430, 795, 253,
    374, 606, 849, 691, 714, 633, 483, 807, 428, 905, 415, 224, 664,
    693, 836, 620, 473, 456, 797, 810, 715, 722, 838, 717, 865, 811,
    607, 913, 723, 697, 378, 436, 818, 320, 622, 813, 485, 431, 839,
    668, 489, 240, 379, 460, 623, 628, 438, 381, 819, 462, 497, 670,
    680, 725, 842, 630, 352, 468, 439, 738, 252, 463, 443, 442, 470,
    248, 684, 843, 739, 900, 671, 784, 850, 821, 729, 929, 792, 368,
    902, 631, 686, 845, 634, 712, 254, 692, 825, 903, 687, 741, 851,
    376, 445, 471, 484, 416, 486, 906, 796, 474, 635, 745, 853, 961,
    866, 694, 798, 907, 716, 808, 475, 637, 695, 255, 718, 576, 914,
    799, 812, 380, 698, 432, 608, 490, 867, 724, 487, 909, 719, 814,
    477, 857, 840, 726, 699, 915, 753, 869, 820, 815, 440, 930, 491,
    624, 672, 740, 917, 464, 844, 382, 498, 931, 822, 727, 962, 873,
    493, 632, 730, 701, 444, 742, 846, 921, 383, 823, 852, 731, 499,
    881, 743, 446, 472, 636, 933, 688, 904, 826, 501, 847, 746, 827,
    733, 447, 963, 937, 476, 854, 868, 638, 908, 488, 696, 747, 829,
    754, 855, 858, 505, 800, 256, 965, 910, 720, 478, 916, 639, 749,
    945, 870, 492, 700, 755, 859, 479, 969, 384, 911, 816, 977, 871,
    918, 728, 494, 874, 702, 932, 757, 861, 500, 732, 824, 923, 875,
    919, 503, 934, 744, 761, 882, 495, 703, 922, 502, 877, 848, 993,
    448, 734, 828, 935, 883, 938, 964, 748, 506, 856, 925, 735, 830,
    966, 939, 885, 507, 750, 946, 967, 756, 860, 941, 831, 912, 872,
    640, 889, 480, 947, 751, 970, 509, 862, 758, 971, 920, 876, 863,
    759, 949, 978, 924, 973, 762, 878, 953, 496, 704, 936, 979, 884,
    763, 504, 926, 879, 736, 994, 886, 940, 995, 981, 927, 765, 942,
    968, 887, 832, 948, 508, 890, 985, 752, 943, 997, 972, 891, 510,
    950, 974, 1001, 893, 951, 864, 760, 1009, 511, 980, 954, 764, 975,
    955, 880, 982, 983, 928, 996, 766, 957, 888, 986, 998, 987, 944,
    892, 999, 767, 512, 989, 1002, 952, 1003, 894, 976, 895, 1010, 956,
    1005, 1011, 958, 984, 959, 988, 1013, 1000, 1017, 768, 990, 1004,
    991, 1006, 960, 1012, 1014, 896, 1007, 1015, 1018, 1019, 992, 1021,
    1008, 1016, 1020, 1022, 1023, 1024]

def SC(N,K,eb_n0):
    snr_dB=eb_n0*(K/N)
    n = int(mt.log2(N))

    #---------finding the Realiability Sequence for the given N------------

    R_Seq_N=[]
    for i in R_Sequence:
        if (i>N):
            continue
        else:
            R_Seq_N.append(i-1)
    # for i in R_Seq_N:
    #     print(i)

    #---------x------x--------x----x-------x------x-----x-----x----x------x

    #---------generating message -------------------

    msg = np.random.randint(2,size=K) # generating message
    #msg=[0, 0, 1, 0]
    # print("message",end="= ")
    # for i in msg:
    #     print(i,end=" ")
    # print("")


    #---------x---------x--------x-------x---------


    #---------setting ui for N-K channels to zero--------

    froz=int(N-K) # generating message
    frozen_pos=R_Seq_N[0:froz] # frozen position
    # print(np.shape(frozen_pos))
    it=0
    ui=[]   #message of length N by adding frozen postion zero

    for i in range(N):
        if i not in frozen_pos:
            ui.append(msg[it])
            it = it + 1
        else:
            ui.append(0)


    # print("ui",end="= ")
    # for p in ui:
    #     print(p,end=" ")
    # print("")
    #-----------x--------x-------x-------x------x------x------



    #-----------encoding it by ui x Gn---------------------------

    # method 2 
    no_bits = 1   # no of bits to encoded at a time 
    for i in range(n):
        for j in range(0, N, 2 * no_bits):
            lf=ui[j:j+no_bits]  
            rt=ui[j+no_bits:j+2*no_bits] 
            ui[j:j+no_bits]=np.add(lf,rt)%2
            ui[j+no_bits:j+2*no_bits]=rt
        no_bits *= 2
    # print("method 2",end="= ")
    # for p in ui:
    #     print(p,end=" ")
    # print("")

    #-----------x-----------x------------------x---------------------x------x



    #------passing the output of the encoder to a BPSK modulator-------------

    bpsk_symbol=np.array(ui)
    bpsk_symbol=1-2*bpsk_symbol # mapping  0--->1 and 1--->-1
    # print("bpsk ",end="= ")
    # for p in bpsk_symbol:
    #     print(p,end=" ")
    # print("")

    #-----------x-----------x------------------x---------------------x------x


    #---------adding AWGN to a signal----------------------------------

    snr_linear = 10 ** (snr_dB / 10) #converting dB to linear
    signal_power =1
    noise_power = signal_power / snr_linear  # calculating noise power
    noise = np.sqrt(noise_power)*np.random.normal(0, 1, bpsk_symbol.shape)  # calculating noise
    noisy_signal = bpsk_symbol + noise   # adding noise to signal 
    # print("noisy signal",end="= ") 
    # for i in noisy_signal:
    #     print(i,end=" ")
    # print("")


    #------------x----------------------x-----------------------x-----------


    #-----------hard decoding- Successive Cancellation Decoder-----------------------------

    def f(a, b):   #min sum
        result = (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.minimum(np.abs(a), np.abs(b))
        return result

    def g(a, b, c):
        result = b + (1 - 2 * c) * a
        return result

    L_values=np.zeros((n+1,N))   #array for storing belief of each node at each depth
    U_values=np.zeros((n+1,N),dtype=int) #array for storing u values of each node at each depth
    node_state=np.zeros(2**(n+1)-1,dtype=int)  # node_state 
    L_values[0, :]=noisy_signal  # adding belief to root node 

    node=0
    depth=0
    done=0

    while done==0:
        npos=int((2**depth)-1+node)  # finding position of node in node state vector
        if depth==n:  # reach leaf node 
            if node in frozen_pos:   #if node is at frozen position
                U_values[n,node]=0 #setting it to 0
            else:
                if(L_values[n,node]>=0): # making decision based belief
                    U_values[n,node]=0
                else:
                    U_values[n,node]=1
            if node==N-1: #reaching last node
                done=1
            else:
                node=mt.floor(node/2) # going to parent
                depth=depth-1
        else:   
            if (node_state[npos]==0):   ## means the node has to go left
                temp=int(2**(n-depth))   
                Inc_belief=L_values[depth,temp*node:temp*(node+1)] #storing incomming belief
                left=Inc_belief[0:temp//2] # storing left half of incomming belief 
                right=Inc_belief[temp//2:] # storing right half of incomming belief
                node=int(node*2)  # going to left
                depth=depth+1  # increasing depth
                temp=int(temp/2)
                L_values[depth,temp*node:temp*(node+1)]=f(left,right) # storing the belief of the left node by calculating min sum
                node_state[npos]=1 # making node state as 1
            elif (node_state[npos]==1):  # means the node has to go right
                temp=int(2**(n-depth))   
                Inc_belief=L_values[depth,temp*node:temp*(node+1)] #storing incomming belief
                left=Inc_belief[0:temp//2]  # storing left half of incomming belief 
                right=Inc_belief[temp//2:]  # storing right half of incomming belief
                lnode=int(2*node)  # storing left node 
                ldepth=depth+1   # storing left node depth
                ltemp=temp//2
                ucapn=U_values[ldepth,ltemp*lnode:ltemp*(lnode+1)]  # storing u values of left node  
                node=int(node*2)+1   #going to right node
                depth=depth+1   # increasing depth
                temp=int(temp/2)
                L_values[depth,temp*node:temp*(node+1)]=g(left,right,ucapn) # storing the belief of the right node by calculating min sum from the left value of u
                node_state[npos]=2 # making node state as 1
            else:  # means the node has to go up
                temp=int(2**(n-depth))   
                lnode=int(2*node)  # storing left node 
                rnode=int(2*node)+1 # storing right node 
                cdepth=depth+1
                ctemp=temp//2
                ucapl=U_values[cdepth,ctemp*lnode:ctemp*(lnode+1)]  # storing u values of left node 
                ucapr=U_values[cdepth,ctemp*rnode:ctemp*(rnode+1)]  # storing u values of left node 
                U_values[depth,temp*node:temp*(node+1)]=np.concatenate((np.add(ucapl,ucapr)%2,ucapr)) # storing u values by  left node u value + right node u value, right node u value 
                node=mt.floor(node/2) # going to parent 
                depth=depth-1   
    msg_c=U_values[n] # storing codeword that are decoded 
    msg_c = np.delete(msg_c, frozen_pos)   # removing the frozen  bits
    # print("final_decoded_mssg",end=" ")
    # for i in final_encoded_mssg:
    #     print(i, end=" ")
    # print(" ")

    # print("initial_mssg",end=" ")
    # for i in inital_message:
    #     print(i, end=" ")
    # print(" ")
    # print(msg_c)
    # print(msg)
    new_dec=((msg_c+msg)%2) #doing xor of mssg and decoded message 
    errors = np.count_nonzero(new_dec == 1) # counting no of one 
    if(errors==0): # represnt no error in the decoded message 
        return 1,errors
    else:
        return 0,errors
 


#----------------------------------------------Main Code runner-------------------------------------------------------Main Code runner-----------------------------------------------------------------------------------#

Prob_success_1 = [] # prob of 1st code word
Prob_success_2 = [] # prob of 2nd code word
Prob_success_3 = [] # prob of 3rd code word
Prob_success_4 = [] # prob of 4th code word

BeR_1 = [] # prob of 1st code word
BeR_2 = [] # prob of 2nd code word
BeR_3 = [] # prob of 3rd code word
BeR_4 = [] # prob of 4th code word

nsim = 1 # no od stimulation

Eb_N0 = np.arange(0,10.5,0.5) #Eb/N0 in dB
N1=[64,128,256,512]  # all N values
K2=[50,100,240,502]  # all K values
for Eb in Eb_N0: #for each Eb/N0 we are calculating success out of nsim
    # print(Eb)
    error_list1=0  
    error_list2=0 
    error_list3=0 
    error_list4=0 
    success_list1=0  
    success_list2=0 
    success_list3=0 
    success_list4=0 
    for j in range(nsim): 
        # print(j)
        success1,val1 = SC(N1[0],K2[0],Eb) 
        success2,val2 = SC(N1[1],K2[1],Eb) 
        success3,val3 = SC(N1[2],K2[2],Eb) 
        success4,val4 = SC(N1[3],K2[3],Eb) 
        error_list1+=val1 # adding 1 if decoding is not successful or 0 id it is successful
        error_list2+=val2 
        error_list3+=val3 
        error_list4+=val4 
        success_list1+=success1
        success_list2+=success2
        success_list3+=success3
        success_list4+=success4
    
    Prob_success_1.append(success_list1/(nsim)) #for each Eb/N0 we are calculating probability of success
    Prob_success_2.append(success_list2/(nsim)) 
    Prob_success_3.append(success_list3/(nsim)) 
    Prob_success_4.append(success_list4/(nsim)) 
    BeR_1.append(error_list1/(nsim*K2[0]))
    BeR_2.append(error_list2/(nsim*K2[1]))
    BeR_3.append(error_list3/(nsim*K2[2]))
    BeR_4.append(error_list4/(nsim*K2[3]))

plt.plot(Eb_N0,Prob_success_1, color='#DAA520',linewidth=1,label = "For N="+ str(N1[0])+ ", K="+str(K2[0])) 
plt.plot(Eb_N0,Prob_success_2,color='#228B22',linewidth=1,label = "For N="+ str(N1[1])+ ", K="+str(K2[1])) 
plt.plot(Eb_N0,Prob_success_3,color='#DC143C',linewidth=1,label = "For N="+ str(N1[2])+ ", K="+str(K2[2])) 
plt.plot(Eb_N0,Prob_success_4,color='#00BFFF',linewidth=1,label = "For N="+ str(N1[3])+ ", K="+str(K2[3])) 

plt.title("Successive Cancellation Decoding for Polar codes") 

plt.xlabel('Eb/N0 (in dB)') 
plt.ylabel('Probability of successful decoding') 
plt.legend() 
plt.grid() 
plt.show() 

plt.plot(Eb_N0,BeR_1, color='#DAA520',linewidth=1,label = "For N="+ str(N1[0])+ ", K="+str(K2[0])) 
plt.plot(Eb_N0,BeR_2, color='#228B22',linewidth=1,label = "For N="+ str(N1[1])+ ", K="+str(K2[1])) 
plt.plot(Eb_N0,BeR_3, color='#DC143C',linewidth=1,label = "For N="+ str(N1[2])+ ", K="+str(K2[2])) 
plt.plot(Eb_N0,BeR_4, color='#00BFFF',linewidth=1,label = "For N="+ str(N1[3])+ ", K="+str(K2[3])) 
plt.title("Successive Cancellation Decoding for Polar codes") 

plt.xlabel('Eb/N0 (in dB)') 
plt.ylabel('BER in dB')
plt.yscale("log") 
plt.legend() 
plt.grid() 
plt.show() 