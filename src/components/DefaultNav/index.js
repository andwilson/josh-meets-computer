import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";
import Img from "gatsby-image";

const Wrapper = styled.div`
  background-color: #28aa55;
  margin-bottom: 18px;
  -webkit-box-shadow: 0px 1px 2px 0px rgba(189,189,189,1);
  -moz-box-shadow: 0px 1px 2px 0px rgba(189,189,189,1);
  box-shadow: 0px 1px 2px 0px rgba(189,189,189,1);
`;

const NavBar = styled.div`
  max-width: 960px;
  flex-wrap: wrap;
  margin: auto;
  padding: 0px 10px 0 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Title = styled.div`
  display: flex;
  > h1 {
    margin: 0;
    line-height: 100%;
  }
`;

const Nav = styled.div`
  display: flex;
  padding: 8px 0 8px 0;
`;

const SLink = styled(Link)`
  font-family: open sans;
  text-decoration: none;
  color: white;
  font-size: 16px;
  margin: 0 0 0 20px;
  transition: all 0.2s ease;
  &:hover {
    color: #e4e4e4;
  }
`;

const TLink = styled(Link)`
  font-family: roboto;
  text-decoration: none;
  color: white;
  font-size: 20px;
  padding: 0;
  margin: 0;
  transition: all 0.2s ease;
  &:hover {
    color: #e4e4e4;
  }
`;

const Boy = styled(Img)`
  margin: 5px 10px 10px 0;
  width: 15px;
`;

const Computer = styled(Img)`
  margin: 5px 0 10px 5px;
  width: 30px;
`;

export default ({ data }) => (
  <Wrapper>
    <NavBar>
      <Title>
        <Boy
          sizes={data.boy.sizes}
        />
        <h1>
          <TLink to={"/"}>{data.site.siteMetadata.title}</TLink>
        </h1>
        <Computer
          sizes={data.computer.sizes}
        />
      </Title>
      <Nav>
        <SLink to={"/projects/"} style={{ marginLeft: 0 }}>
          Projects
        </SLink>
        <SLink to={"/notes/"}>Notes</SLink>
        <SLink to={"/letters/"}>Letters</SLink>
        <SLink to={"/about/"}>About</SLink>
      </Nav>
    </NavBar>
  </Wrapper>
);
