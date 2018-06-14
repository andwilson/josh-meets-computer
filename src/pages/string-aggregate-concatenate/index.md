---
title: "String Aggregate And Concatenate"
date: "2018-06-14"
path: "/string-aggregate-concatenate/"
category: "Notes"
section: "SQL"
---

There has been a few times where I had to build a query in such a way that I needed to both aggregate groups of data while showing information of each grouped line. In Manufacturing, one specific report was a production dashboard, which summarized all the operations of a production job (total run time, total lead time, number of operations), but also needed to show the "routing" of the job, which was the name of each operation being grouped together. The output needed to be one line per job. The solution was to concatenate the names of each operation into one row per group.

Another problem I had months later was a Purchase Price variance report for Finance. They wanted one line per Purchase Order, showing the variance that occured against the paid invoice. This was a problem because there were some instances of multiple invoices making partial payments to a single Purchase Order, duplicating the Purchase order line which was displaying the total purchase price. The solution was to group the invoices by Purchase Order, but we didn't want to lose out on the invoice numbers. The solution was to list the related invoices in a string. 

In both of these scenarios, it was useful to aggregate a character field into a comma seperated list for grouping. Below is how to do so in both PostgreSQL and SQL Server.


```python
%sql postgresql://Joshua:Sparky80@localhost:5432/pgguide
```




    'Connected: Joshua@pgguide'



## Understanding the Dataset
I will be using a demo dataset that can be downloaded from the PostgreSQL website. It has three tables: 

- Products: holding a table of products for sale
- Purchases: holding a list of users who've made purchases
- Purchase_Items: holding a list of items purchased



```python
%%sql postgresql://
    
    SELECT * FROM products limit 5;
```

    5 rows affected.
    




<table>
    <tr>
        <th>id</th>
        <th>title</th>
        <th>price</th>
        <th>created_at</th>
        <th>deleted_at</th>
        <th>tags</th>
    </tr>
    <tr>
        <td>1</td>
        <td>Dictionary</td>
        <td>9.99</td>
        <td>2011-01-01 10:00:00-10:00</td>
        <td>None</td>
        <td>[&#x27;Book&#x27;]</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Python Book</td>
        <td>29.99</td>
        <td>2011-01-01 10:00:00-10:00</td>
        <td>None</td>
        <td>[&#x27;Book&#x27;, &#x27;Programming&#x27;, &#x27;Python&#x27;]</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Ruby Book</td>
        <td>27.99</td>
        <td>2011-01-01 10:00:00-10:00</td>
        <td>None</td>
        <td>[&#x27;Book&#x27;, &#x27;Programming&#x27;, &#x27;Ruby&#x27;]</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Baby Book</td>
        <td>7.99</td>
        <td>2011-01-01 10:00:00-10:00</td>
        <td>None</td>
        <td>[&#x27;Book&#x27;, &#x27;Children&#x27;, &#x27;Baby&#x27;]</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Coloring Book</td>
        <td>5.99</td>
        <td>2011-01-01 10:00:00-10:00</td>
        <td>None</td>
        <td>[&#x27;Book&#x27;, &#x27;Children&#x27;]</td>
    </tr>
</table>




```python
%%sql postgresql://
    
    SELECT * FROM purchases limit 5;
```

    5 rows affected.
    




<table>
    <tr>
        <th>id</th>
        <th>created_at</th>
        <th>name</th>
        <th>address</th>
        <th>state</th>
        <th>zipcode</th>
        <th>user_id</th>
    </tr>
    <tr>
        <td>1</td>
        <td>2011-03-16 05:03:00-10:00</td>
        <td>Harrison Jonson</td>
        <td>6425 43rd St.</td>
        <td>FL</td>
        <td>50382</td>
        <td>7</td>
    </tr>
    <tr>
        <td>2</td>
        <td>2011-09-13 19:00:00-10:00</td>
        <td>Cortney Fontanilla</td>
        <td>321 MLK Ave.</td>
        <td>WA</td>
        <td>43895</td>
        <td>30</td>
    </tr>
    <tr>
        <td>3</td>
        <td>2011-09-10 19:54:00-10:00</td>
        <td>Ruthie Vashon</td>
        <td>2307 45th St.</td>
        <td>GA</td>
        <td>98937</td>
        <td>18</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2011-02-27 10:53:00-10:00</td>
        <td>Isabel Wynn</td>
        <td>7046 10th Ave.</td>
        <td>NY</td>
        <td>57243</td>
        <td>11</td>
    </tr>
    <tr>
        <td>5</td>
        <td>2011-12-20 02:45:00-10:00</td>
        <td>Shari Dutra</td>
        <td>4046 8th Ave.</td>
        <td>FL</td>
        <td>61539</td>
        <td>34</td>
    </tr>
</table>




```python
%%sql postgresql://
    
    SELECT * FROM purchase_items limit 5;
```

    5 rows affected.
    




<table>
    <tr>
        <th>id</th>
        <th>purchase_id</th>
        <th>product_id</th>
        <th>price</th>
        <th>quantity</th>
        <th>state</th>
    </tr>
    <tr>
        <td>2</td>
        <td>1</td>
        <td>3</td>
        <td>27.99</td>
        <td>1</td>
        <td>Delivered</td>
    </tr>
    <tr>
        <td>3</td>
        <td>1</td>
        <td>8</td>
        <td>108.00</td>
        <td>1</td>
        <td>Delivered</td>
    </tr>
    <tr>
        <td>4</td>
        <td>2</td>
        <td>1</td>
        <td>9.99</td>
        <td>2</td>
        <td>Delivered</td>
    </tr>
    <tr>
        <td>5</td>
        <td>3</td>
        <td>12</td>
        <td>9.99</td>
        <td>1</td>
        <td>Delivered</td>
    </tr>
    <tr>
        <td>6</td>
        <td>3</td>
        <td>17</td>
        <td>14.99</td>
        <td>4</td>
        <td>Delivered</td>
    </tr>
</table>



## Grouping Possibilities
As shown below, there are buyers that have bought multiple items. In this demo, we want to show the buyers name, the total amount he paid and what items he bought, with one buyer per row. This requires that we aggregate the Product field, which is a category type column.


```python
%%sql postgresql://
            
            SELECT 
                purch.Name as "Buyer", 
                prod.title as "Product", 
                items.price * items.quantity as "Cost",
                RANK() OVER(
                    PARTITION BY items.purchase_id
                    ORDER BY items.product_id) AS "Item"
                
            FROM purchase_items items
            INNER JOIN products prod ON
                items.product_id = prod.id
            INNER JOIN purchases purch ON
                items.purchase_id = purch.id
            
            limit 10;

```

    10 rows affected.
    




<table>
    <tr>
        <th>Buyer</th>
        <th>Product</th>
        <th>Cost</th>
        <th>Item</th>
    </tr>
    <tr>
        <td>Harrison Jonson</td>
        <td>Ruby Book</td>
        <td>27.99</td>
        <td>1</td>
    </tr>
    <tr>
        <td>Harrison Jonson</td>
        <td>MP3 Player</td>
        <td>108.00</td>
        <td>2</td>
    </tr>
    <tr>
        <td>Cortney Fontanilla</td>
        <td>Dictionary</td>
        <td>19.98</td>
        <td>1</td>
    </tr>
    <tr>
        <td>Ruthie Vashon</td>
        <td>Classical CD</td>
        <td>9.99</td>
        <td>1</td>
    </tr>
    <tr>
        <td>Ruthie Vashon</td>
        <td>Holiday CD</td>
        <td>9.99</td>
        <td>2</td>
    </tr>
    <tr>
        <td>Ruthie Vashon</td>
        <td>Documentary</td>
        <td>59.96</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Isabel Wynn</td>
        <td>Baby Book</td>
        <td>23.97</td>
        <td>1</td>
    </tr>
    <tr>
        <td>Shari Dutra</td>
        <td>Python Book</td>
        <td>119.96</td>
        <td>1</td>
    </tr>
    <tr>
        <td>Shari Dutra</td>
        <td>Romantic</td>
        <td>14.99</td>
        <td>2</td>
    </tr>
    <tr>
        <td>Kristofer Galvez</td>
        <td>Coloring Book</td>
        <td>5.99</td>
        <td>1</td>
    </tr>
</table>



## PostgreSQL Implementation

Turns out it is extremely easy to do so in postgreSQL, because there is a built in function called "STRING_AGG". This does all the heavy lifting and easily concatenates the strings for us.


```python
%%sql postgresql://

        SELECT 
            purch.Name as "Buyer", 
            SUM(items.price * items.quantity) as "Total Paid",
            STRING_AGG(prod.title, ' | ') as "Products Purchased",
            count(purch.Name) as "Item Count"

        FROM purchase_items items
        INNER JOIN products prod ON
            prod.id = items.product_id
        INNER JOIN purchases purch ON
            purch.id = items.purchase_id

        GROUP BY purch.Name
        ORDER BY "Item Count" DESC
        limit 10
```

    10 rows affected.
    




<table>
    <tr>
        <th>Buyer</th>
        <th>Total Paid</th>
        <th>Products Purchased</th>
        <th>Item Count</th>
    </tr>
    <tr>
        <td>Evelyn Fretz</td>
        <td>72.93</td>
        <td>Holiday CD | Dictionary | Action | Baby Book</td>
        <td>4</td>
    </tr>
    <tr>
        <td>Russ Petrin</td>
        <td>539.95</td>
        <td>Pop CD | Electronic CD | Desktop Computer | Classical CD</td>
        <td>4</td>
    </tr>
    <tr>
        <td>Williams Selden</td>
        <td>544.95</td>
        <td>Dictionary | Drama | Desktop Computer | Classical CD</td>
        <td>4</td>
    </tr>
    <tr>
        <td>Angel Coderre</td>
        <td>1050.96</td>
        <td>Action | 42&quot; LCD TV | Baby Book</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Andres Schippers</td>
        <td>2539.96</td>
        <td>42&quot; LCD TV | Country CD | Action</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Allen Harshberger</td>
        <td>209.89</td>
        <td>Electronic CD | Action | Ruby Book</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Alfonzo Jay</td>
        <td>1038.98</td>
        <td>Desktop Computer | Pop CD | 42&quot; Plasma TV</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Alfonzo Haubrich</td>
        <td>2359.93</td>
        <td>Comedy Movie | Laptop Computer | Desktop Computer</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Alfonzo Bodkin</td>
        <td>3721.91</td>
        <td>Classical CD | Laptop Computer | Ruby Book</td>
        <td>3</td>
    </tr>
    <tr>
        <td>Adell Mayon</td>
        <td>1408.98</td>
        <td>Dictionary | 42&quot; LCD TV | Laptop Computer</td>
        <td>3</td>
    </tr>
</table>



## SQL Server Implementation

In SQL Server, it's a bit harder. One suggestion online is to use the "FOR XML PATH" function, which concatenates the field into an XML path by a delimiter. This is packaged in the "STUFF" function, which stuffs one string into another string and removes the first comma for us.

I do not have access to a SQL Server where my jupyter notebook is stored, but the accompanied repository has a DDL for the data which can be pasted into SQL Fiddle along with the code below.


```python
%%sql sqlserver://
    
        WITH Purch AS (
          SELECT 
            purchasers.name,
            items.product_id as "product_id",
            round(items.price * items.quantity, 2) as "Total",
            products.title

          FROM purchase_items items
            
          INNER JOIN purchases purchasers ON
            purchasers.id = items.purchase_id
            
          INNER JOIN products ON
            products.id = items.product_id
        )

        SELECT 
          Purch2.name,
          STUFF((
            SELECT ', ' + Purch1.title
            FROM Purch AS Purch1
            WHERE Purch1.name = Purch2.name
            FOR XML PATH ('')),1 ,1 , '') AS [Products],
          sum(Purch2.Total) AS [Total Paid]
        FROM Purch AS Purch2
        GROUP BY Purch2.Name


```

An alternative is to use the PIVOT function if you know the maximum number of lines in each grouping. This works if you don't have too many lines in each group. In this case there are ever only four items bought, so we Pivot the rank function (which counts number of rows in a group) into columns, and have a case when function to return the columns concatenated into a string.


```python
        WITH Purch AS (
          SELECT 
            purchasers.name,
            items.product_id as "product_id",
            round(items.price * items.quantity, 2) as "Total",
            products.title,
            RANK() OVER(
                PARTITION BY
                    purchasers.name
                ORDER BY
                    items.product_id) as "Count"

          FROM purchase_items items
          INNER JOIN purchases purchasers ON
            purchasers.id = items.purchase_id
          INNER JOIN products ON
            products.id = items.product_id
        )

        SELECT
            Purch.name,
            Purch.product_id,
            Purch.Total,
            Purch.title,
            (
            case when Purch.1 is null then '' else Purch.1 end +
            case when Purch.2 is null then '' else ', ' + Purch.2 +
            case when Purch.3 is null then '' else ', ' + Purch.3 +
            case when Purch.4 is null then '' else ', ' + Purch.4 end end end) as "Purchased Items"
        FROM Purch
        PIVOT
            max(Purch.title) for Purch.Count in ([1],[2],[3], [4])
            
                
        
```
