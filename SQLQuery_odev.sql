--1. SORU (veriyi import flat file yoluyla aldým, ID ekledim)
ALTER TABLE FLO
ADD ID INT IDENTITY(1,1) PRIMARY KEY

SELECT * FROM FLO
--2. SORU
SELECT COUNT(DISTINCT master_id) unique_customer_count
FROM FLO

--3. SORU
SELECT
	SUM(order_num_total_ever_online + order_num_total_ever_offline) order_num_total_ever_online,
	SUM(customer_value_total_ever_offline + customer_value_total_ever_online) customer_value_total_ever
FROM FLO

--4. SORU
SELECT
	SUM(customer_value_total_ever_offline + customer_value_total_ever_online)/
	NULLIF(SUM(order_num_total_ever_online + order_num_total_ever_offline), 0)
	AS basket_price
FROM FLO

--5. SORU

SELECT
last_order_channel,
	SUM(order_num_total_ever_online + order_num_total_ever_offline) order_num_total_ever_online,
	SUM(customer_value_total_ever_offline + customer_value_total_ever_online) customer_value_total_ever
FROM FLO
GROUP BY last_order_channel


--6. SORU
SELECT
store_type,
	SUM(customer_value_total_ever_offline + customer_value_total_ever_online) customer_value_total_ever
FROM FLO
GROUP BY store_type


--7. SORU
SELECT
YEAR(first_order_date) year,
	SUM(order_num_total_ever_online + order_num_total_ever_offline) order_num_total_ever_online
FROM FLO
GROUP BY YEAR(first_order_date)

--8. SORU
SELECT
last_order_channel,
	SUM(customer_value_total_ever_offline + customer_value_total_ever_online)/
	NULLIF(SUM(order_num_total_ever_online + order_num_total_ever_offline), 0)
	AS basket_price
FROM FLO
GROUP BY last_order_channel


--9. SORU Son 12 ayda en çok ilgi gören kategoriyi getiren sorguyu yazýnýz.

SELECT DISTINCT interested_in_categories_12 FROM FLO


SELECT TOP 5 value AS category, COUNT(*) AS interest_count
FROM FLO
CROSS APPLY STRING_SPLIT(
       REPLACE(REPLACE(REPLACE(interested_in_categories_12,'[',''),']',''),' ','')
       ,','
)
WHERE last_order_date >= DATEADD(MONTH,-12,(SELECT MAX(last_order_date) FROM FLO))
GROUP BY value
ORDER BY interest_count DESC



--10. SORU En çok tercih edilen store_type bilgisini getiren sorguyu yazýnýz

SELECT store_type FROM FLO



SELECT TOP 1 TRIM(value) AS storetype, COUNT(*) AS interest_count
FROM FLO
CROSS APPLY string_split(store_type,',')
GROUP BY value
ORDER BY interest_count DESC


--11. SORU En son alýþveriþ yapýlan kanal (last_order_channel) bazýnda, en çok ilgi gören kategoriyi ve bu kategoriden ne kadarlýk alýþveriþ yapýldýðýný getiren sorguyu yazýnýz.

SELECT last_order_channel, TRIM(value) AS category,
	COUNT(*) AS interest_count, 
	SUM(customer_value_total_ever_offline + customer_value_total_ever_online) TOTALSALE
FROM FLO
CROSS APPLY string_split(
       REPLACE(REPLACE(REPLACE(interested_in_categories_12,'[',''),']',''),' ','')
       ,',')
	   WHERE value <> ''
GROUP BY last_order_channel, value
ORDER BY last_order_channel, interest_count DESC


--12. SORU En çok alýþveriþ yapan kiþinin ID’ sini getiren sorguyu yazýnýz.
SELECT * FROM FLO

SELECT TOP 1 master_id, SUM(customer_value_total_ever_offline + customer_value_total_ever_online) TOTALSALE
FROM FLO
GROUP BY master_id
ORDER BY TOTALSALE DESC


--13. SORU  En çok alýþveriþ yapan kiþinin alýþveriþ baþýna ortalama cirosunu ve alýþveriþ yapma gün ortalamasýný (alýþveriþ sýklýðýný) getiren sorguyu yazýnýz.

SELECT TOP 1
	master_id,
	(customer_value_total_ever_offline + customer_value_total_ever_online)
		/ NULLIF(order_num_total_ever_offline + order_num_total_ever_online, 0) basket,
	DATEDIFF(DAY, first_order_date, last_order_date)
	/ NULLIF(order_num_total_ever_online + order_num_total_ever_offline - 1, 0) avg_freq
FROM FLO

ORDER BY order_num_total_ever_offline + order_num_total_ever_online DESC


--14. SORU  En çok alýþveriþ yapan (ciro bazýnda) ilk 100 kiþinin alýþveriþ yapma gün ortalamasýný (alýþveriþ sýklýðýný) getiren sorguyu yazýnýz

SELECT AVG(CAST(DATEDIFF(DAY, first_order_date, last_order_date) AS FLOAT) 
           / NULLIF(total_orders - 1, 0)) AS avg_freq
FROM (
    SELECT TOP 100
        first_order_date,
        last_order_date,
        order_num_total_ever_online + order_num_total_ever_offline AS total_orders
    FROM FLO
    ORDER BY customer_value_total_ever_online + customer_value_total_ever_offline DESC
) temp


--15. SORU En son alýþveriþ yapýlan kanal (last_order_channel) kýrýlýmýnda en çok alýþveriþ yapan müþteriyi getiren sorguyu yazýnýz.

SELECT
    last_order_channel,
    master_id,
    (order_num_total_ever_online + order_num_total_ever_offline) AS total_order
FROM FLO
WHERE (order_num_total_ever_online + order_num_total_ever_offline) =
(
    SELECT MAX(order_num_total_ever_online + order_num_total_ever_offline)
    FROM FLO flo2
    WHERE flo2.last_order_channel = FLO.last_order_channel
)


--16. SORU  En son alýþveriþ yapan kiþinin ID’ sini getiren sorguyu yazýnýz. (Max son tarihte birden fazla alýþveriþ yapan ID bulunmakta. Bunlarý da getiriniz.)

--1.
SELECT TOP 1 WITH TIES
master_id, last_order_date
FROM FLO
ORDER BY last_order_date DESC

--2.
SELECT master_id, last_order_date
FROM FLO
WHERE last_order_date = (SELECT MAX(last_order_date) FROM FLO);
