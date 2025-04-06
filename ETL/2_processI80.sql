SET @group_num := 0;

WITH calculations AS (
    SELECT 
        CONCAT(CAST(S.Vehicle_ID AS CHAR), "_", CAST(S.`Preceding` AS CHAR)) AS Pair_ID,
        S.Vehicle_ID AS Subject, 
        S.Space_Headway AS Subject_Space_Headway, 
        S.v_Vel AS Subject_Velocity,
        (S.v_Vel - L.v_Vel) AS Subject_Delta_Velocity,
        S.v_Acc AS Subject_Acceleration,
        L.Vehicle_ID AS Leader,
        L.Space_Headway AS Leader_Space_Headway,
        L.v_Vel AS Leader_Velocity,
        (L.v_Vel - XL.v_Vel) AS Leader_Delta_Velocity,
        L.v_Acc AS Leader_Acceleration, 
        XL.Vehicle_ID AS LeadLeader,
        XL.v_Vel AS LL_Velcocity,
        S.Global_Time AS Global_Time,
        LAG(S.Global_Time) OVER (PARTITION BY S.Vehicle_ID, S.`Preceding` ORDER BY S.Global_Time) AS Previous_Time,
        (S.Global_Time - LAG(S.Global_Time) OVER (PARTITION BY S.Vehicle_ID, S.`Preceding` ORDER BY S.Global_Time)) AS Time_Diff,
        IF((S.Global_Time - LAG(S.Global_Time) OVER (PARTITION BY S.Vehicle_ID, S.`Preceding` ORDER BY S.Global_Time)) = 100, "Consecutive", "Not_Consecutive") AS Time_Condition
    FROM 
        ngsim_i_80 S
    JOIN 
        (SELECT * FROM ngsim_i_80) L 
        ON S.`Preceding` = L.Vehicle_ID
    JOIN 
        (SELECT * FROM ngsim_i_80) XL 
        ON L.`Preceding` = XL.Vehicle_ID  
    WHERE 
        S.Lane_ID = L.Lane_ID 
        AND L.Lane_ID = XL.Lane_ID
        AND S.Space_Headway > 0
        AND L.Space_Headway > 0
        AND XL.Space_Headway > 0
        AND S.Space_Headway <= 150
        AND L.Space_Headway <= 150
        AND S.Global_Time = L.Global_Time
        AND L.Global_Time = XL.Global_Time
        AND S.`Preceding` != 0
        AND L.`Preceding` != 0
        AND XL.`Preceding` != 0
        AND L.`Following` = S.Vehicle_ID
        AND XL.`Following` = L.Vehicle_ID
    ORDER BY 
        Pair_ID, S.Global_Time ASC
),

grouped_calculations AS (
    SELECT
        c.*,
        @group_num := IF(c.Time_Condition = 'Not_Consecutive', @group_num + 1, @group_num) AS Group_Num
    FROM 
        calculations c
),

grouped_counts AS (
    SELECT 
        gc.Pair_ID, 
        gc.Group_Num, 
        COUNT(*) AS count_rows
    FROM 
        grouped_calculations gc
    GROUP BY 
        gc.Pair_ID, 
        gc.Group_Num
    HAVING 
        COUNT(*) > 100
)

SELECT 
    gc.*, 
    CONCAT(gc.Pair_ID, '_', gc.Group_Num) AS Group_ID
INTO OUTFILE 'ProcessI80.csv'
FROM 
    grouped_calculations gc
JOIN 
    grouped_counts gc_counts 
    ON gc.Pair_ID = gc_counts.Pair_ID 
    AND gc.Group_Num = gc_counts.Group_Num
ORDER BY 
    gc.Pair_ID, 
    gc.Global_Time ASC;
