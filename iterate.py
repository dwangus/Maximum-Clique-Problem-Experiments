import math

global spacing
spacing = 5000.0

def iterate(n):
    a = [0.0,0.0,0.0]
    b = [spacing,0.0,0.0]
    c = [spacing/2.0, (spacing/2.0)*math.sqrt(3), 0.0]
    d = [spacing/2.0, float((spacing/2.0)/math.sqrt(3)), float(spacing*math.sqrt(2.0/3.0))]
    points = {"A":a, "B":b, "C":c, "D":d}

    #a-b, b-c, a-c, a-d
    
    for iteration in range(n):
        update = {}
        for point in points.keys():
            if point == "A":
                AB = (points["B"][0] - points["A"][0], points["B"][1] - points["A"][1], points["B"][2] - points["A"][2])
                AC = (points["C"][0] - points["A"][0], points["C"][1] - points["A"][1], points["C"][2] - points["A"][2])
                AD = (points["D"][0] - points["A"][0], points["D"][1] - points["A"][1], points["D"][2] - points["A"][2])
                vec_A = (AB[0] + AC[0] + AD[0], AB[1] + AC[1] + AD[1], AB[2] + AC[2] + AD[2])
                mag_A = math.sqrt(vec_A[0]**2 + vec_A[1]**2 + vec_A[2]**2)
                u_A = (vec_A[0]/mag_A, vec_A[1]/mag_A, vec_A[2]/mag_A)
                update["A"] = u_A
            elif point == "B":
                BA = (points["A"][0] - points["B"][0], points["A"][1] - points["B"][1], points["A"][2] - points["B"][2])
                BC = (points["C"][0] - points["B"][0], points["C"][1] - points["B"][1], points["C"][2] - points["B"][2])
                vec_B = (BA[0] + BC[0], BA[1] + BC[1], BA[2] + BC[2])
                mag_B = math.sqrt(vec_B[0]**2 + vec_B[1]**2 + vec_B[2]**2)
                u_B = (vec_B[0]/mag_B, vec_B[1]/mag_B, vec_B[2]/mag_B)
                update["B"] = u_B
            elif point == "C":
                CA = (points["A"][0] - points["C"][0], points["A"][1] - points["C"][1], points["A"][2] - points["C"][2])
                CB = (points["B"][0] - points["C"][0], points["B"][1] - points["C"][1], points["B"][2] - points["C"][2])
                vec_C = (CA[0] + CB[0], CA[1] + CB[1], CA[2] + CB[2])
                mag_C = math.sqrt(vec_C[0]**2 + vec_C[1]**2 + vec_C[2]**2)
                u_C = (vec_C[0]/mag_C, vec_C[1]/mag_C, vec_C[2]/mag_C)
                update["C"] = u_C
            else:
                vec_D = (points["A"][0] - points["D"][0], points["A"][1] - points["D"][1], points["A"][2] - points["D"][2])
                mag_D = math.sqrt(vec_D[0]**2 + vec_D[1]**2 + vec_D[2]**2)
                u_D = (vec_D[0]/mag_D, vec_D[1]/mag_D, vec_D[2]/mag_D)
                update["D"] = u_D
        for change in update.keys():
            for x in range(3):
                points[change][x] += update[change][x]
    print("Iterations: {0}\n Node Positions: {1}".format(n, points))
    dAB = math.sqrt((points["A"][0] - points["B"][0])**2 + (points["A"][1] - points["B"][1])**2 + (points["A"][2] - points["B"][2])**2)
    print("AB Distance: {0}".format(dAB))
    dAC = math.sqrt((points["A"][0] - points["C"][0])**2 + (points["A"][1] - points["C"][1])**2 + (points["A"][2] - points["C"][2])**2)
    print("AC Distance: {0}".format(dAC))
    dAD = math.sqrt((points["A"][0] - points["D"][0])**2 + (points["A"][1] - points["D"][1])**2 + (points["A"][2] - points["D"][2])**2)
    print("AD Distance: {0}".format(dAD))
    dBC = math.sqrt((points["C"][0] - points["B"][0])**2 + (points["C"][1] - points["B"][1])**2 + (points["C"][2] - points["B"][2])**2)
    print("BC Distance: {0}".format(dBC))
iterate(100)
iterate(500)
iterate(1982)
iterate(1983)
iterate(3200)
